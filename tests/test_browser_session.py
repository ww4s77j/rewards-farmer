"""What a started Edge session is told about timeouts and prompts.

A run whose page stopped answering spent twenty minutes in WebDriver waits
because nothing bounded a page load and an unexpected dialog could block every
following command. These pin down the settings that make it fail in bounded
time instead.

None of them start a browser.

	python -m unittest discover -s tests
"""

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, SRC)

import accounts
import browser

ACCOUNT = accounts.get_default_account()


class BrowserSessionSettings(unittest.TestCase):
	def setUp(self):
		edge = browser.webdriver.Edge
		self.addCleanup(setattr, browser.webdriver, "Edge", edge)

	def test_options_dismiss_an_unexpected_prompt_instead_of_waiting(self):
		options = browser.build_options(ACCOUNT, headless=False)

		self.assertEqual(options.to_capabilities().get("unhandledPromptBehavior"), "dismiss and notify")

	def test_a_started_driver_gets_a_page_load_timeout(self):
		seen = {}

		class FakeDriver:
			def set_page_load_timeout(self, seconds):
				seen["page_load_timeout"] = seconds

		browser.webdriver.Edge = lambda *args, **kwargs: FakeDriver()

		driver = browser.start_driver(ACCOUNT)

		self.assertIsInstance(driver, FakeDriver)
		self.assertEqual(seen["page_load_timeout"], 60)


if __name__ == "__main__":
	unittest.main()
