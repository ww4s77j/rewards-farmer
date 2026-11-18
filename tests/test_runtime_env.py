"""Runtime settings loaded from dotenv after import, without browser or API calls."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import accounts
import dotenv


def fresh_module(name):
	spec = importlib.util.spec_from_file_location("runtime_env_" + name, SRC / (name + ".py"))
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


class RuntimeEnvTests(unittest.TestCase):
	def setUp(self):
		self.stack = self.enterContext(mock.patch.dict(os.environ, {}, clear=True))
		self.temp = self.enterContext(tempfile.TemporaryDirectory())
		self.env_path = Path(self.temp) / ".env"

	def load_settings(self, text):
		self.env_path.write_text(text, encoding="utf-8")
		dotenv.load_dotenv(self.env_path)

	def options(self, module, **kwargs):
		with mock.patch.object(module.webdriver, "Edge", side_effect=AssertionError("Browser forbidden")):
			if hasattr(module, "edge_version"):
				with mock.patch.object(module.edge_version, "installed_major", return_value=157):
					return module.build_options(accounts.Account("synthetic", "synthetic-profile", "Default"), **kwargs)
			return module.build_options(accounts.Account("synthetic", "synthetic-profile", "Default"), **kwargs)

	def test_dotenv_loaded_after_browser_import_controls_built_options(self):
		module = fresh_module("browser")
		self.load_settings("REWARDS_HEADLESS=true\n")
		self.assertIn("--headless=new", self.options(module).arguments)

	def test_explicit_visible_overrides_dotenv_hidden(self):
		module = fresh_module("browser")
		self.load_settings("REWARDS_HEADLESS=true\n")
		self.assertNotIn("--headless=new", self.options(module, headless=False).arguments)

	def test_docker_environment_before_import_and_truthy_spellings_still_work(self):
		for value in ("true", "1", "yes"):
			with self.subTest(value=value):
				os.environ["REWARDS_HEADLESS"] = value
				self.assertIn("--headless=new", self.options(fresh_module("browser")).arguments)

	def test_environment_change_between_builds_is_not_cached(self):
		module = fresh_module("browser")
		os.environ["REWARDS_HEADLESS"] = "true"
		self.assertIn("--headless=new", self.options(module).arguments)
		os.environ["REWARDS_HEADLESS"] = "false"
		self.assertNotIn("--headless=new", self.options(module).arguments)

	def test_llm_provider_from_dotenv_after_import_selects_matching_request_parts(self):
		module = fresh_module("llm_utils")
		self.load_settings("LLM_PROVIDER=openrouter\nOPENROUTER_BASE_URL=http://synthetic.invalid/v1\nOPENROUTER_MODEL=synthetic/model:free\nOPENROUTER_API_KEY=synthetic-not-a-key\n")
		with mock.patch.object(module.requests, "post", side_effect=AssertionError("Provider requests forbidden")):
			self.assertEqual(module._get_llm_base_url(), "http://synthetic.invalid/v1")
			self.assertEqual(module._get_llm_model(), "synthetic/model:free")
			self.assertEqual(module._get_llm_headers()["Authorization"], "Bearer synthetic-not-a-key")

	def test_openrouter_model_default_is_free(self):
		module = fresh_module("llm_utils")
		os.environ.update(LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="synthetic")
		self.assertEqual(module._get_llm_model(), "openrouter/free")

	def test_openrouter_paid_model_is_refused(self):
		module = fresh_module("llm_utils")
		os.environ.update(
			LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="synthetic",
			OPENROUTER_MODEL="openai/gpt-4o-mini",
		)
		with self.assertRaises(RuntimeError) as caught:
			module._get_llm_model()
		self.assertIn("free", str(caught.exception).lower())

	def test_openrouter_free_suffix_is_accepted(self):
		module = fresh_module("llm_utils")
		os.environ.update(
			LLM_PROVIDER="openrouter", OPENROUTER_API_KEY="synthetic",
			OPENROUTER_MODEL="google/gemma-4-31b-it:free",
		)
		self.assertEqual(module._get_llm_model(), "google/gemma-4-31b-it:free")

	def test_llm_provider_can_switch_back_to_local_after_import(self):
		os.environ["LLM_PROVIDER"] = "openrouter"
		module = fresh_module("llm_utils")
		os.environ.update(LLM_PROVIDER="local", LOCAL_LLM_BASE_URL="http://synthetic-local.invalid/v1", LOCAL_LLM_MODEL="synthetic-local")
		self.assertEqual(module._get_llm_base_url(), "http://synthetic-local.invalid/v1")
		self.assertEqual(module._get_llm_model(), "synthetic-local")
		self.assertNotIn("Authorization", module._get_llm_headers())

	def test_existing_deepseek_key_alone_selects_deepseek(self):
		module = fresh_module("llm_utils")
		os.environ["DEEPSEEK_API_KEY"] = "synthetic-deepseek-key"
		self.assertEqual(module._get_llm_base_url(), "https://api.deepseek.com")
		self.assertEqual(module._get_llm_model(), "deepseek-chat")
		self.assertEqual(module._get_llm_headers()["Authorization"], "Bearer synthetic-deepseek-key")

	def test_deepseek_provider_from_dotenv_after_import(self):
		module = fresh_module("llm_utils")
		self.load_settings("LLM_PROVIDER=deepseek\nDEEPSEEK_BASE_URL=http://synthetic-deepseek.invalid\nDEEPSEEK_MODEL=synthetic-deepseek\nDEEPSEEK_API_KEY=synthetic-not-a-key\n")
		self.assertEqual(module._get_llm_base_url(), "http://synthetic-deepseek.invalid")
		self.assertEqual(module._get_llm_model(), "synthetic-deepseek")
		self.assertEqual(module._get_llm_headers()["Authorization"], "Bearer synthetic-not-a-key")

	def test_explicit_deepseek_without_key_fails_before_request(self):
		module = fresh_module("llm_utils")
		os.environ["LLM_PROVIDER"] = "deepseek"
		with self.assertRaisesRegex(RuntimeError, "DEEPSEEK_API_KEY"):
			module._get_llm_headers()


if __name__ == "__main__":
	unittest.main()
