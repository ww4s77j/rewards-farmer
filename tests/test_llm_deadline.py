"""A model request may not park a run indefinitely.

An OpenRouter connection stayed open without answering and held the search
task for minutes even though a 60 s socket timeout was set. Each attempt now
has a hard wall-clock deadline, there are three attempts in total, and then
the caller falls back to its local generator.

None of these touch the network.

	python -m unittest discover -s tests
"""

import importlib.util
import os
import sys
import time
import unittest
from unittest import mock

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)


def fresh_module(name):
	spec = importlib.util.spec_from_file_location("llm_deadline_" + name, os.path.join(SRC, name + ".py"))
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	return module


class LlmDeadline(unittest.TestCase):
	def setUp(self):
		self.module = fresh_module("llm_utils")

	def test_a_hung_request_is_given_up_after_three_bounded_attempts(self):
		calls = []

		def hung(messages):
			calls.append(1)
			time.sleep(5)

		self.module.get_llm_response = hung

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.1):
			started = time.monotonic()

			with self.assertRaises(RuntimeError):
				self.module.get_nonempty_llm_response([])

			self.assertLess(time.monotonic() - started, 3)

		self.assertEqual(len(calls), 3)

	def test_a_later_attempt_can_still_answer(self):
		answers = [RuntimeError("first attempt failed"), "a query"]

		def flaky(messages):
			answer = answers.pop(0)

			if isinstance(answer, Exception):
				raise answer

			return answer

		self.module.get_llm_response = flaky

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.5):
			self.assertEqual(self.module.get_nonempty_llm_response([]), "a query")

	def test_empty_answers_share_the_same_three_attempts(self):
		calls = []

		def empty(messages):
			calls.append(1)
			return "  "

		self.module.get_llm_response = empty

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.5):
			with self.assertRaises(RuntimeError):
				self.module.get_nonempty_llm_response([])

		self.assertEqual(len(calls), 3)


class LlmQueryValidation(unittest.TestCase):
	"""A free model once returned its whole reasoning chain as the content, and
	the search box received all of it. That must count as a failed attempt."""

	BLOB = (
		"how to test water quality at home\n\n"
		"we now have a chain of 7 queries, but the00000\n\n"
		"the user wants me to continue the chain with a new query\n"
	)

	def setUp(self):
		self.module = fresh_module("llm_utils")

	def test_a_reasoning_blob_is_rejected_and_tried_three_times(self):
		calls = []

		def blob(messages):
			calls.append(1)
			return self.BLOB

		self.module.get_llm_response = blob

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.5):
			with self.assertRaises(RuntimeError):
				self.module.get_query_response([])

		self.assertEqual(len(calls), 3)

	def test_a_later_attempt_can_return_a_usable_query(self):
		answers = [self.BLOB, "  Water   Filter Types  "]

		self.module.get_llm_response = lambda messages: answers.pop(0)

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.5):
			self.assertEqual(self.module.get_query_response([]), "water filter types")

	def test_a_long_single_line_is_rejected(self):
		self.module.get_llm_response = lambda messages: "word " * 40

		with mock.patch.object(self.module, "LLM_TIMEOUT_SECONDS", 0.5):
			with self.assertRaises(RuntimeError):
				self.module.get_query_response([])

	def test_the_task_query_falls_back_when_the_blob_never_improves(self):
		self.module.get_query_response = lambda messages: (_ for _ in ()).throw(RuntimeError("blob"))

		query = self.module.get_search_query_from_task_description("Search on Bing to find water filters")

		self.assertLessEqual(len(query), self.module.QUERY_MAX_CHARS)
		self.assertNotIn("\n", query)


if __name__ == "__main__":
	unittest.main()
