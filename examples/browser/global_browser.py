import asyncio
import os
from pathlib import Path

from playwright.async_api import Page, async_playwright

from metrics.metrics import metrics_counter_inc
from neo.utils import logger

_BEDROCK_PROJECT = os.environ.get("BEDROCK_PROJECT", "")


def is_bedrock_env() -> bool:
    return _BEDROCK_PROJECT != ""


async def handle_new_page(page: Page):
    """
    Handle new page events and execute custom logic
    """
    print(f"New page created: {page.url}")


async def launch_chrome_debug(use_chrome_channel: bool = False, headless: bool = False):
    """
    Launch Chrome browser with remote debugging enabled on port 9222
    Returns the browser instance when launched successfully
    """
    try:
        extension_path = Path(os.path.dirname(__file__)).joinpath("browser_extension/error_capture")  # type: ignore
        playwright = await async_playwright().start()
        disable_security_args = [
            "--disable-web-security",
            "--disable-site-isolation-trials",
            "--disable-features=IsolateOrigins,site-per-process",
        ]
        workspace = "/workspace" if is_bedrock_env() else "./workspace"
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=os.path.join(workspace, "browser", "user_data"),
            headless=headless,
            viewport={"width": 1280, "height": 720},
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                "--disable-background-timer-throttling",
                "--disable-popup-blocking",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-window-activation",