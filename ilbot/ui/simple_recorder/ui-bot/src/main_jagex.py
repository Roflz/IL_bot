import win32gui

from jagex_automation import (
    open_jagex_in_browser,
    check_for_cloudflare,
    navigate_to_account_creation,
    automate_account_creation,
    get_verification_code_from_gmail,
    enter_verification_code,
    enter_jagex_account_name,
    create_jagex_password,
    complete_registration,
    navigate_and_click_characters_pyautogui, launch_jagex_launcher, login_to_jagex_launcher,
    enter_launcher_verification_code, find_text_in_window, process_character_list, create_account_from_jagex_launcher
)
import pyautogui
import time

def account_profile_page():
    """
    Check if the current Chrome window is on the Jagex account profile page.

    Returns:
        bool: True if we're on the profile page
    """
    return find_text_in_window("Google Chrome", "accountjagexcom/en-GB/manage/profile")

def main(email: str):
    print("🚀 Jagex Account Creation with PyAutoGUI")
    print("=" * 50)

    create_new_jagex_account = False
    create_characters_from_account = True

    def print_all_windows():
        def enum_handler(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_title = win32gui.GetWindowText(hwnd)
                if window_title:
                    windows.append(window_title)
            return True

        windows = []
        win32gui.EnumWindows(enum_handler, windows)
        for title in windows:
            print(title)

    if create_new_jagex_account:
        launch_jagex_launcher()

        create_account_from_jagex_launcher()

    if create_characters_from_account:

        open_jagex_in_browser()

        if account_profile_page():
            # Navigate to account profile and click Characters button using PyAutoGUI
            navigate_and_click_characters_pyautogui()

            print("\n🎉 Account creation and navigation process completed!")

            launch_jagex_launcher()

            login_to_jagex_launcher("charlieeeeee54345@gmail.com")

            process_character_list()

if __name__ == "__main__":
    email = "mahnriley+test@gmail.com"
    main(email)