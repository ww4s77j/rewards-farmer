"""Skips for work that is already done or that cannot pay out.

A run used to reopen every daily set activity, search every Explore on Bing
card and click app-install promos such as "Download the Bing app now to claim
500 points as a new user!", even when there was nothing left to earn.

None of these need a browser.

	python -m unittest discover -s tests
"""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from selenium.webdriver.common.by import By

from element_selectors import ElementSelectionUtils
from fakes import FakeDriver, FakeElement
import rewards_tasks


def make_tasks():
	"""A RewardsTaskUtils without the browser its __init__ opens."""
	tasks = rewards_tasks.RewardsTaskUtils.__new__(rewards_tasks.RewardsTaskUtils)
	tasks.driver = FakeDriver()
	tasks.driver.current_window_handle = "main"
	tasks.tab_utils = types.SimpleNamespace(
		close_all_other_tabs=lambda *a, **k: None,
		close_all_other_tabs_except=lambda *a, **k: None,
		switch_to_other_tab=lambda: None,
		ensure_focus=lambda: None,
	)
	tasks.mouse = types.SimpleNamespace(
		wheel_scroll_element_into_view=lambda element: None,
		wheel_scroll_to_top=lambda: None,
	)
	tasks.keyboard = types.SimpleNamespace(send_keys=lambda keys: None)
	tasks.move_to_and_click = mock.Mock()
	tasks.wait_for_element = mock.Mock(side_effect=lambda getter, timeout=10: getter() if callable(getter) else getter)
	tasks.wait_for_then_click = lambda getter, timeout=10: tasks.move_to_and_click(getter)
	tasks.switch_to_earn_page = lambda: None
	tasks.progress = ""
	return tasks


def selectors_with_daily_tile(text):
	row = FakeElement(children={(By.TAG_NAME, "button"): [
		FakeElement(text="Bing App Streak\nCheck-in: 0/1"),
		FakeElement(text=text),
	]})
	return ElementSelectionUtils(FakeDriver(children={(By.ID, "streaks"): [row]}))


def card(description="", href=None, status=None):
	children = {}
	if description:
		children[(By.CSS_SELECTOR, "p:nth-child(2)")] = [FakeElement(text=description)]
	if status is not None:
		status_children = {(By.TAG_NAME, "p"): [FakeElement(text=status)]}
		children[(By.CSS_SELECTOR, "div.flex.w-full.items-center.gap-2")] = [FakeElement(text=status, children=status_children)]
	attributes = {"href": href} if href else {}
	return FakeElement(attributes=attributes, children=children)


class DailySetSkip(unittest.TestCase):
	def test_complete_tile_is_not_opened(self):
		tasks = make_tasks()
		tasks.elements = selectors_with_daily_tile("Daily Set Streak\nActivity: 3/3")
		tasks.elements.get_daily_set_elements = mock.Mock(return_value=[])

		with mock.patch.object(rewards_tasks.time, "sleep", lambda seconds: None):
			tasks.complete_bing_daily_set()

		tasks.move_to_and_click.assert_not_called()

	def test_incomplete_tile_still_opens_every_activity(self):
		tasks = make_tasks()
		tasks.elements = selectors_with_daily_tile("Daily Set Streak\nActivity: 1/3")
		tasks.elements.get_daily_set_elements = mock.Mock(return_value=[FakeElement(), FakeElement(), FakeElement()])
		tasks.elements.get_daily_set_element_by_index = lambda index: FakeElement()

		with mock.patch.object(rewards_tasks.time, "sleep", lambda seconds: None):
			tasks.complete_bing_daily_set()

		self.assertEqual(tasks.move_to_and_click.call_count, 4)  # opener plus three activities


class ExploreSkip(unittest.TestCase):
	def test_only_incomplete_cards_are_searched(self):
		tasks = make_tasks()
		complete = card(description="Search on Bing", status="Completed")
		open_card = card(description="Search for the weather", status="3 points")
		tasks.elements = ElementSelectionUtils(FakeDriver())
		tasks.elements.get_explore_on_bing_elements = mock.Mock(return_value=[complete, open_card])
		tasks.wait_for_element = mock.Mock(return_value=FakeElement())
		tasks.elements.get_bing_search_bar = lambda: FakeElement()

		with mock.patch.object(rewards_tasks.queries, "search_query_for_task", lambda description: "weather"), \
			mock.patch.object(rewards_tasks.time, "sleep", lambda seconds: None):
			tasks.complete_explore_on_bing_tasks()

		tasks.move_to_and_click.assert_called_once_with(open_card)

	def test_all_complete_cards_are_not_searched(self):
		tasks = make_tasks()
		tasks.elements = ElementSelectionUtils(FakeDriver())
		tasks.elements.get_explore_on_bing_elements = mock.Mock(return_value=[
			card(description="Search on Bing", status="Completed"),
		])
		tasks.wait_for_element = mock.Mock(return_value=FakeElement())

		tasks.complete_explore_on_bing_tasks()

		tasks.move_to_and_click.assert_not_called()


class AppPromo(unittest.TestCase):
	def setUp(self):
		self.elements = ElementSelectionUtils(FakeDriver())

	def test_download_the_bing_app_card_is_a_promo(self):
		observed = "Download the Bing app now to claim 500 points as a new user!"
		self.assertTrue(self.elements.card_is_app_promo(card(description=observed)))

	def test_bingapp_host_is_a_promo(self):
		self.assertTrue(self.elements.card_is_app_promo(card(href="https://bingapp.microsoft.com/install")))

	def test_ordinary_cards_are_not_promos(self):
		for description in ("Search on Bing", "Turn referrals into rewards", "Complete 3 searches to earn 10 points"):
			with self.subTest(description=description):
				self.assertFalse(self.elements.card_is_app_promo(card(description=description)))

	def test_misc_cards_click_the_normal_card_only(self):
		tasks = make_tasks()
		tasks.elements = ElementSelectionUtils(FakeDriver())
		promo = card(description="Download the Bing app now to claim 500 points as a new user!", status="500 points")
		normal = card(description="Take a quiz", status="10 points")
		tasks.elements.get_all_misc_cards = lambda: [promo, normal]

		with mock.patch.object(rewards_tasks.time, "sleep", lambda seconds: None):
			tasks.complete_misc_cards()

		tasks.move_to_and_click.assert_called_once_with(normal)


if __name__ == "__main__":
	unittest.main()
