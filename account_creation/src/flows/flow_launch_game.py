from time import sleep
from typing import Optional
from pywinauto.application import Application

def launch_from_launcher(
    window_title_substring: str,
    button_text: Optional[str] = None,
    click_coords: Optional[tuple[int,int]] = None
):
    app = Application(backend="uia").connect(title_re=f".*{window_title_substring}.*", timeout=10)
    dlg = app.window(title_re=f".*{window_title_substring}.*")
    dlg.set_focus()
    dlg.set_focus()
    sleep(0.5)

    if button_text:
        btn = dlg.child_window(title=button_text, control_type="Button")
        btn.wait("enabled", timeout=10)
        btn.click_input()
    elif click_coords:
        dlg.click_input(coords=click_coords)
    else:
        raise ValueError("Provide either button_text or click_coords to trigger launch.")
