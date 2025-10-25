# UI Bot Scaffold

Automates:
1) Web login
2) Email OTP retrieval (IMAP or Gmail API)
3) Character creation via web UI
4) Windows game launcher click

## Quick start
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

# Set env as needed in .env (see src/config.py for keys)
python -m src.main

## Notes
- Creating new consumer Gmail accounts via API is not supported. Use an existing mailbox,
  a custom domain inbox, or Google Workspace Admin SDK (for org users) if applicable.
- Email polling: choose either IMAP (simpler) or Gmail API (OAuth).
- Update selectors in src/selectors/example_selectors.py to match your target site.
- Update launcher window title/button text in flow_launch_game.py for your environment.
