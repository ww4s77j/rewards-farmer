import os
import re
from typing import Generator
import logging
import random
import requests
import threading
import os
from constants import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_FOR_SEARCH_QUEST = (
	"You are a helpful assistant tasked with creating a search query based on a directive. "
	"Output nothing but the search query you create, and do not include any additional commentary or explanation. "
	"Do not include any labels or quotes. "
	"The search query must be the only output, and do not format the query as an imperative to 'search for' something. "
	"Imagine that your output will be fed directly into a search engine as you provide it. "
	"For example, if the directive is 'Search on Bing for the latest news about space exploration', you might output 'latest news space exploration'. "
	"Outputting 'search on Bing for the latest news about space exploration' or 'search bing.com/news for space exploration' would be incorrect, "
	"as those answers include instructions to perform a search rather than just the search query itself. "
	"Additionally, be specific, e.g. if a prompt asks you to search for vacation flights, include "
	"a specific destination rather than just searching 'vacation flights'. The current year is 2026. "
	"Make your query concise, ideally 6 words or less, and do not include any punctuation. "
	"Write it the way a person types into a search box, not the way an assistant phrases a request. "
	"Never use location placeholders such as 'near me', 'near you' or 'in my area'. "
)

DEFAULT_USER_PROMPT_FOR_SEARCH_QUEST_WITHOUT_DESC = """Base your search query on the following task description: """

DEFAULT_SYSTEM_PROMPT_FOR_SEARCH_POINTS = (
	"The user is interested in learning more about topics related to a word that will be given to you. "
	"Your task is to come up with subsequent search queries that relate to each other, each one branching out "
	"from the previous one so that the user can explore a topic in depth. Your first search query should be "
	"based on the word that the user gives you, and each subsequent search query should be at least remotely based on the previous ones. "
	"Output only the single search query you come up with and do not include any additional commentary or explanation. Do not include any labels or quotes. "
	"The search queries should ideally be short (6 words max) and do not need to be fully fledged questions, but they should be unique. The current year is 2026. "
	"Write each query exactly as a person would type it into a search box: plain wording, varied phrasing between queries, "
	"and no location placeholders such as 'near me', 'near you' or 'in my area'. "
	"Never add quotes, labels, instructions, or marketing phrasing like 'best deals' unless the topic itself calls for it."
)

DEFAULT_USER_PROMPT_FOR_SEARCH_POINTS_WITHOUT_DESC = """Generate the first search query based on the following word: """

USER_PROMPT_FOR_SEARCH_QUERY_CONTINUATION = """Generate the next search query."""

# A connection that stays open without answering is not covered by a socket
# read timeout, so each attempt also has a wall-clock deadline. Three attempts
# cover a transient failure; after that the caller's local fallback takes over.
LLM_TIMEOUT_SECONDS = 30
LLM_ATTEMPTS = 3

# A search query is short. Anything longer, or with line breaks in it, is the
# model's reasoning rather than an answer; a free model has returned its whole
# chain of thought as the content and the search box received all of it.
QUERY_MAX_CHARS = 80

DEFAULT_LLM_PROVIDER = "local"
PREFERRED_LLM_PROVIDER = "deepseek"

def _get_llm_provider() -> str:
	explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
	if explicit:
		return explicit

	# With no explicit choice, use DeepSeek whenever its key is available so an
	# exported DEEPSEEK_API_KEY is enough to switch providers.
	if os.getenv("DEEPSEEK_API_KEY", "").strip():
		return PREFERRED_LLM_PROVIDER

	return DEFAULT_LLM_PROVIDER

def _get_llm_base_url() -> str:
	provider = _get_llm_provider()
	if provider == "deepseek":
		return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
	elif provider in ("openrouter", "open-router"):
		return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
	elif provider in ("local", "ollama"):
		if base := os.getenv("LOCAL_LLM_BASE_URL", "").strip():
			return base.rstrip("/")
		host = os.getenv("OLLAMA_HOST", "localhost:11434").strip()
		if not host.startswith(("http://", "https://")):
			host = f"http://{host}"
		return f"{host.rstrip('/')}/v1"

	raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Supported values are 'deepseek', 'openrouter' and 'local'.")

def _is_free_openrouter_model(model: str) -> bool:
	"""OpenRouter marks free variants with a `:free` suffix; its router is free too."""
	return model == "openrouter/free" or model.endswith(":free")

def _get_llm_model() -> str:
	provider = _get_llm_provider()
	if provider == "deepseek":
		return os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip()
	elif provider in ("openrouter", "open-router"):
		model = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()

		if not _is_free_openrouter_model(model):
			raise RuntimeError(
				"OPENROUTER_MODEL must name a free model, either 'openrouter/free' "
				f"or an id ending in ':free'; refusing {model!r}"
			)

		return model
	elif provider in ("local", "ollama"):
		return os.getenv("LOCAL_LLM_MODEL", "gemma4:cloud").strip()

	raise ValueError(f"Unsupported LLM_PROVIDER: {provider}. Supported values are 'deepseek', 'openrouter' and 'local'.")

def _get_llm_headers() -> dict[str, str]:
	headers = {
		"Content-Type": "application/json",
	}

	if _get_llm_provider() == "deepseek":
		api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
		if not api_key:
			raise RuntimeError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek")

		headers["Authorization"] = f"Bearer {api_key}"
		return headers

	if _get_llm_provider() in {"openrouter", "open-router"}:
		api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
		if not api_key:
			raise RuntimeError("OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter")

		headers["Authorization"] = f"Bearer {api_key}"
		referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
		title = os.getenv("OPENROUTER_TITLE", "").strip()

		if referer:
			headers["HTTP-Referer"] = referer
		if title:
			headers["X-Title"] = title
		return headers

	api_key = os.getenv("LOCAL_LLM_API_KEY", "").strip()
	if api_key:
		headers["Authorization"] = f"Bearer {api_key}"

	return headers

def _request_timeout() -> float:
	return float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "") or LLM_TIMEOUT_SECONDS)

def _with_deadline(call, seconds: float):
	"""Run call in a daemon thread and give up after seconds, whatever it is doing.

	A daemon thread rather than an executor: a request that never returns must
	not keep the process alive at exit, and the thread is abandoned, not joined.
	"""
	result: dict[str, object] = {}

	def run():
		try:
			result["value"] = call()
		except Exception as exc:
			result["error"] = exc

	thread = threading.Thread(target=run, daemon=True)
	thread.start()
	thread.join(seconds)

	if thread.is_alive():
		raise TimeoutError(f"LLM request did not answer within {seconds:g}s")

	if "error" in result:
		raise result["error"]

	return result.get("value")

def get_llm_response(messages: list[dict[str, str]]) -> str:
	timeout = _request_timeout()
	response = requests.post(
		f"{_get_llm_base_url()}/chat/completions",
		headers=_get_llm_headers(),
		json={
			"model": _get_llm_model(),
			"messages": messages,
		},
		timeout=(min(10.0, timeout), timeout),
	)
	response.raise_for_status()

	content = response.json()["choices"][0]["message"]["content"]

	if not isinstance(content, str):
		raise RuntimeError(f"Unexpected LLM response content type: {type(content)!r}")

	return content

def get_nonempty_llm_response(messages: list[dict[str, str]]) -> str:
	"""Three bounded attempts in total, then the caller's local fallback."""
	last = "no attempt made"

	for attempt in range(1, LLM_ATTEMPTS + 1):
		try:
			response = _with_deadline(lambda: get_llm_response(messages), _request_timeout())
		except Exception as exc:
			last = exc
			logger.warning("LLM request failed (%s/%s): %s", attempt, LLM_ATTEMPTS, exc)
			continue

		if response and response.strip():
			return response

		last = "empty response"
		logger.warning("Empty LLM response (%s/%s)", attempt, LLM_ATTEMPTS)

	raise RuntimeError(f"LLM returned nothing usable after {LLM_ATTEMPTS} attempts: {last}")

def get_query_response(messages: list[dict[str, str]]) -> str:
	"""Three bounded attempts, and each answer has to look like one search query.

	`get_nonempty_llm_response` accepts any non-empty text, which is how a
	reasoning model's chain of thought ended up typed into Bing: it is
	non-empty, it is just not an answer.
	"""
	last = "no attempt made"

	for attempt in range(1, LLM_ATTEMPTS + 1):
		try:
			response = _with_deadline(lambda: get_llm_response(messages), _request_timeout())
		except Exception as exc:
			last = exc
			logger.warning("LLM request failed (%s/%s): %s", attempt, LLM_ATTEMPTS, exc)
			continue

		query = " ".join((response or "").split())

		if query and len(query) <= QUERY_MAX_CHARS:
			return query.lower()

		last = f"unusable response of {len(query)} characters"
		logger.warning(
			"LLM returned something that is not a search query (%s/%s): %s characters",
			attempt, LLM_ATTEMPTS, len(query)
		)

	raise RuntimeError(f"LLM returned no usable search query after {LLM_ATTEMPTS} attempts: {last}")

def get_search_query_from_task_description(task_description: str) -> str:
	# compat
	if "lyrics of your favorite song" in task_description.lower(): return "sweet caroline lyrics"

	messages = [
		{
			"role": "system",
			"content": DEFAULT_SYSTEM_PROMPT_FOR_SEARCH_QUEST
		},
		{
			"role": "user",
			"content": DEFAULT_USER_PROMPT_FOR_SEARCH_QUEST_WITHOUT_DESC + task_description
		}
	]

	try:
		response = get_query_response(messages)
		return response.lower()
	except Exception as exc:
		logger.warning("LLM is offline or unavailable (%s). Using fallback search query generator.", exc)
		words = [w for w in re.sub(r"[^\w\s]", "", task_description).split() if len(w) > 3 and w.lower() not in {"search", "bing", "find", "about", "with", "from", "that", "this"}]
		fallback_query = " ".join(words[:4]) if words else f"{get_random_noun()} search"
		return fallback_query.lower()


def get_related_search_queries(seed_word: str, num_queries: int=20) -> Generator[str, None, None]:
	messages = [
		{
			"role": "system",
			"content": DEFAULT_SYSTEM_PROMPT_FOR_SEARCH_POINTS
		},
		{
			"role": "user",
			"content": DEFAULT_USER_PROMPT_FOR_SEARCH_POINTS_WITHOUT_DESC + seed_word
		}
	]

	use_fallback = False

	for _ in range(num_queries):
		if not use_fallback:
			try:
				response = get_query_response(messages)
				yield response.lower()

				messages.append({
					"role": "assistant",
					"content": response
				})

				messages.append({
					"role": "user",
					"content": USER_PROMPT_FOR_SEARCH_QUERY_CONTINUATION
				})
				continue
			except Exception as exc:
				logger.warning("LLM is offline or unavailable (%s). Using built-in generator for remaining queries.", exc)
				use_fallback = True

		noun1 = get_random_noun()
		noun2 = get_random_noun()
		yield f"{seed_word} {noun1} {noun2}".lower()


NOUNS = [
	noun.strip().lower() for noun in open(os.path.join(REPO_ROOT, "nouns.txt"), "r", encoding="utf-8").read().splitlines()
	if len(noun.strip()) >= 3
]


def get_random_noun() -> str:
	return random.choice(NOUNS)
