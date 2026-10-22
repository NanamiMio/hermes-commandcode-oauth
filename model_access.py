"""Account model access rules observed in the installed Command Code 1.87.1 client."""
import time
import urllib.parse

# This order is the CLI's billing order, not a guess from model names or prices.
PLAN_ORDER = (
    "individual-go", "individual-go-v1", "individual-goat", "individual-pro",
    "individual-pro-v1", "individual-provider", "individual-max", "individual-ultra", "teams-pro",
)
ACTIVE_STATUSES = {"active", "trialing", "past_due"}


def organization_id(whoami):
    if not isinstance(whoami, dict):
        return ""
    nested = whoami.get("org")
    values = [nested.get("id") if isinstance(nested, dict) else None]
    values.extend(whoami.get(key) for key in ("organizationId", "orgId", "organization_id"))
    return next((v.strip() for v in values if isinstance(v, str) and v.strip()), "")


def fetch_access_context(origin, token, get_json, *, deadline):
    """Share the catalog's timeout budget; an incomplete billing lookup grants no lock."""
    def get(path):
        remaining = deadline - time.monotonic()
        return get_json(origin + path, token, timeout=remaining) if remaining > 0 else None

    whoami = get("/alpha/whoami")
    if not isinstance(whoami, dict):
        return None
    org = organization_id(whoami)
    suffix = "?orgId=" + urllib.parse.quote(org, safe="") if org else ""
    subscription = get("/alpha/billing/subscriptions" + suffix)
    credit_response = get("/alpha/billing/credits" + suffix)
    if not isinstance(subscription, dict) or not subscription.get("success") or not isinstance(credit_response, dict):
        return None
    data, credits = subscription.get("data"), credit_response.get("credits")
    if not isinstance(data, dict) or not isinstance(credits, dict) or data.get("status") not in ACTIVE_STATUSES:
        return None
    # Unknown plans are deliberately advisory, as in the CLI.
    if data.get("planId") not in PLAN_ORDER:
        return None
    for field in ("purchasedCredits", "freeCredits"):
        value = credits.get(field, 0)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
    return {"plan": data["planId"], "extra": any(credits.get(k, 0) > 0 for k in ("purchasedCredits", "freeCredits"))}


def access_note(model, context):
    """Empty means selectable or unknown; a nonempty note means a confirmed plan lock."""
    category = model.get("category")
    if not context or context["extra"] or not isinstance(category, str) or category not in {"opensource", "premium"}:
        return ""
    minimum = model.get("minimumPlan")
    if "minimumPlan" in model and minimum is None:
        return "Locked: extra usage credits required"
    if minimum in PLAN_ORDER:
        if PLAN_ORDER.index(context["plan"]) < PLAN_ORDER.index(minimum):
            return f"Locked: requires {minimum} or extra usage credits"
        return ""
    # Old catalogs without a known minimum retain the CLI's category rule.
    if context["plan"] in PLAN_ORDER[:3] and model.get("category") == "premium":
        return "Locked: requires Pro or extra usage credits"
    return ""


def model_rows(payload, context=None):
    if not isinstance(payload, dict):
        return None
    raw = payload.get("data") if "data" in payload else payload.get("models")
    if not isinstance(raw, list):
        return None
    result, seen = [], set()
    for item in raw:
        item = item if isinstance(item, dict) else {"id": item}
        name = item.get("id")
        if not isinstance(name, str) or not name.strip() or name.strip() in seen:
            continue
        name = name.strip()
        seen.add(name)
        if item.get("hidden") is True:
            continue
        note = access_note(item, context)
        label = item.get("name")
        label = label.strip() if isinstance(label, str) and label.strip() else name
        result.append({"id": name, "label": label, "available": not bool(note),
                       **({"note": note} if note else {})})
    return result if result or not raw or seen else None
