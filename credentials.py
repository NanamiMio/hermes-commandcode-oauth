"""Read credentials in the active Hermes profile; standalone callers use their env."""
import os


def env_token() -> str:
    try:
        from agent.secret_scope import get_secret_str
    except ModuleNotFoundError as exc:
        if exc.name not in ("agent", "agent.secret_scope"):
            raise
        return os.environ.get("COMMANDCODE_CLI_TOKEN", "").strip()
    return get_secret_str("COMMANDCODE_CLI_TOKEN").strip()


def allow_cli_fallback() -> bool:
    try:
        from agent.secret_scope import serves_routed_profile
    except ModuleNotFoundError as exc:
        if exc.name not in ("agent", "agent.secret_scope"):
            raise
        return True
    # A served profile must opt into the CLI account through auth add, not inherit it.
    return not serves_routed_profile()
