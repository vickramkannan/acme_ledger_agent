import os
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Acme Appliances Ledger Agent")
LOCAL_TZ = ZoneInfo("Asia/Kolkata")
USD_PER_UNIT = {"USD": 1.0, "EUR": 1.03, "INR": 0.01208}

LEDGER_API_URL = os.getenv("LEDGER_API_URL", "").strip()
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN", "").strip()
AI_PIPE_BASE_URL = os.getenv(
    "AI_PIPE_BASE_URL", "https://aipipe.org/openrouter/v1"
).rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-4.1-nano")

class QuestionRequest(BaseModel):
    question: str

def api_url(path: str, page=None):
    if not LEDGER_API_URL:
        raise RuntimeError("LEDGER_API_URL environment variable is not set")
    parts = urlparse(LEDGER_API_URL)
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    params["path"] = path
    if page is not None:
        params["page"] = str(page)
    else:
        params.pop("page", None)
    return urlunparse(parts._replace(query=urlencode(params)))

def fetch_ledger():
    # Fetch page 1, then every remaining page using the API's page count.
    first = requests.get(api_url("/orders", 1), timeout=12)
    first.raise_for_status()
    payload = first.json()
    pages = int(payload.get("pages", 1))
    records = list(payload.get("orders", []))
    for page in range(2, pages + 1):
        response = requests.get(api_url("/orders", page), timeout=12)
        response.raise_for_status()
        records.extend(response.json().get("orders", []))

    # Keep the latest version of each order ID.
    current = {}
    for row in records:
        order_id = str(row.get("id", ""))
        previous = current.get(order_id)
        if previous is None or parse_dt(row.get("updated_at")) > parse_dt(previous.get("updated_at")):
            current[order_id] = row

    rate_response = requests.get(api_url("/rates"), timeout=12)
    rate_response.raise_for_status()
    rate_data = rate_response.json().get("usd_per_unit", USD_PER_UNIT)
    return list(current.values()), rate_data

def parse_dt(value):
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def local_date(value):
    return parse_dt(value).astimezone(LOCAL_TZ).date().isoformat()

def usd_amount(row, rates):
    currency = str(row.get("currency", "USD")).upper()
    return float(row.get("amount", 0) or 0) * float(rates.get(currency, USD_PER_UNIT.get(currency, 1.0)))

def get_plan(question):
    """Ask the model to translate natural language into a small, auditable query plan."""
    if not AI_PIPE_TOKEN:
        return basic_plan(question)

    system = """
You convert questions about an order ledger into a JSON query plan. Return ONLY valid JSON.
Schema:
{
 "measure":"revenue|count|quantity|average_order_value|orders|customers",
 "filters":{"region":string|null,"product":string|null,"customer":string|null,
 "status":string|null,"currency":string|null,"date_from":"YYYY-MM-DD"|null,
 "date_to":"YYYY-MM-DD"|null,"date_field":"created_at|updated_at"},
 "group_by":"region|product|customer|status|currency|month|none",
 "sort":"asc|desc",
 "limit":integer|null,
 "metric_currency":"USD|original"
}
Rules:
- "revenue", sales, turnover, money earned => measure revenue and status paid unless user explicitly asks about another status.
- For "orders" / order details, measure orders; for number/how many use count; quantity/units sold use quantity.
- Dates refer to Asia/Kolkata business dates. date_from and date_to are inclusive.
- Use latest/current records only; this is already handled by the service.
- Revenue is counted only for paid orders. If they ask about refunded/void orders, set that status explicitly.
- If user asks for top/best/highest, sort desc; bottom/lowest sort asc.
- Use group_by only when comparing/breaking down by a field; otherwise none.
- If they request a currency conversion or monetary total, metric_currency USD unless they explicitly request original currency.
- Do not invent filters. Unknown filters must be null.
"""
    user = f"Question: {question}\nReturn the JSON query plan."
    try:
        resp = requests.post(
            AI_PIPE_BASE_URL + "/chat/completions",
            headers={"Authorization": f"Bearer {AI_PIPE_TOKEN}", "Content-Type": "application/json"},
            json={
                "model": AI_MODEL,
                "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}
            },
            timeout=10,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        plan = json.loads(content)
        return normalize_plan(plan, question)
    except Exception:
        return basic_plan(question)

def normalize_plan(plan, question):
    filters = plan.get("filters") or {}
    normalized = {
        "measure": plan.get("measure", "count"),
        "filters": {
            "region": filters.get("region"),
            "product": filters.get("product"),
            "customer": filters.get("customer"),
            "status": filters.get("status"),
            "currency": filters.get("currency"),
            "date_from": filters.get("date_from"),
            "date_to": filters.get("date_to"),
            "date_field": filters.get("date_field", "created_at"),
        },
        "group_by": plan.get("group_by", "none"),
        "sort": plan.get("sort", "desc"),
        "limit": plan.get("limit"),
        "metric_currency": plan.get("metric_currency", "USD"),
    }
    normalized["measure"] = normalized["measure"] if normalized["measure"] in {
        "revenue", "count", "quantity", "average_order_value", "orders", "customers"
    } else "count"
    if normalized["measure"] == "revenue" and not normalized["filters"]["status"] and not re.search(r"\b(refund|refunded|void)\b", question, re.I):
        normalized["filters"]["status"] = "paid"
    if normalized["group_by"] not in {"region", "product", "customer", "status", "currency", "month", "none"}:
        normalized["group_by"] = "none"
    return normalized

def basic_plan(question):
    q = question.lower()
    plan = {
        "measure": "count",
        "filters": {"region": None, "product": None, "customer": None, "status": None,
                    "currency": None, "date_from": None, "date_to": None, "date_field": "created_at"},
        "group_by": "none", "sort": "desc", "limit": None, "metric_currency": "USD"
    }
    for region in ["north", "south", "east", "west", "central"]:
        if re.search(r"\b" + region + r"\b", q):
            plan["filters"]["region"] = region.title()
    products = ["Rice Cooker", "Air Fryer", "Blender", "Mixer", "Juicer", "Grinder", "Toaster", "Kettle"]
    for product in products:
        if product.lower() in q:
            plan["filters"]["product"] = product
    customer = re.search(r"\bC\d{4}\b", question, re.I)
    if customer:
        plan["filters"]["customer"] = customer.group().upper()
    for status in ["paid", "refunded", "void"]:
        if status in q:
            plan["filters"]["status"] = status
    if any(word in q for word in ["revenue", "sales", "turnover", "earned", "sales value"]):
        plan["measure"] = "revenue"
        if not plan["filters"]["status"]:
            plan["filters"]["status"] = "paid"
    elif any(word in q for word in ["quantity", "units sold", "number of units"]):
        plan["measure"] = "quantity"
    elif any(word in q for word in ["average order value", "average value"]):
        plan["measure"] = "average_order_value"
        plan["filters"]["status"] = plan["filters"]["status"] or "paid"
    elif "customer" in q and any(word in q for word in ["how many", "number of", "count"]):
        plan["measure"] = "customers"
    elif any(word in q for word in ["list", "show me", "which orders", "order details"]):
        plan["measure"] = "orders"
    if "by region" in q or "per region" in q:
        plan["group_by"] = "region"
    elif "by product" in q or "per product" in q:
        plan["group_by"] = "product"
    elif "by status" in q:
        plan["group_by"] = "status"
    elif "by currency" in q:
        plan["group_by"] = "currency"
    elif "by customer" in q or "per customer" in q:
        plan["group_by"] = "customer"
    if "top " in q or "highest" in q or "most " in q or "best " in q:
        plan["sort"] = "desc"
        m = re.search(r"\btop\s+(\d+)", q)
        if m:
            plan["limit"] = int(m.group(1))
    if "lowest" in q or "bottom " in q or "least " in q:
        plan["sort"] = "asc"
    return plan

def filter_rows(rows, plan):
    f = plan["filters"]
    out = []
    for row in rows:
        if f["region"] and str(row.get("region", "")).lower() != f["region"].lower():
            continue
        if f["product"] and str(row.get("product", "")).lower() != f["product"].lower():
            continue
        if f["customer"] and str(row.get("customer", "")).upper() != f["customer"].upper():
            continue
        if f["status"] and str(row.get("status", "")).lower() != f["status"].lower():
            continue
        if f["currency"] and str(row.get("currency", "")).upper() != f["currency"].upper():
            continue
        date_field = f.get("date_field") or "created_at"
        date_value = local_date(row.get(date_field))
        if f["date_from"] and date_value < f["date_from"]:
            continue
        if f["date_to"] and date_value > f["date_to"]:
            continue
        out.append(row)
    return out

def row_metric(row, measure, rates, metric_currency):
    if measure == "revenue":
        if metric_currency == "original":
            return float(row.get("amount", 0) or 0)
        return usd_amount(row, rates)
    if measure == "quantity":
        return float(row.get("qty", 0) or 0)
    return 1.0

def answer_plan(rows, rates, plan):
    filtered = filter_rows(rows, plan)
    measure = plan["measure"]
    group_by = plan["group_by"]
    currency_mode = plan.get("metric_currency", "USD")

    if measure == "orders":
        result = [{
            "id": r.get("id"), "customer": r.get("customer"), "region": r.get("region"),
            "product": r.get("product"), "qty": r.get("qty"), "amount": r.get("amount"),
            "currency": r.get("currency"), "status": r.get("status"),
            "created_at": r.get("created_at"), "updated_at": r.get("updated_at")
        } for r in filtered]
        return {"count": len(result), "orders": result[:int(plan.get("limit") or 100)]}

    if measure == "customers":
        customers = sorted({r.get("customer") for r in filtered if r.get("customer")})
        return {"count": len(customers), "customers": customers[:int(plan.get("limit") or 100)]}

    if measure == "average_order_value":
        paid = [r for r in filtered if str(r.get("status", "")).lower() == "paid"]
        total = sum(usd_amount(r, rates) for r in paid)
        return {"count": len(paid), "average_order_value_usd": round(total / len(paid), 2) if paid else 0.0}

    def value_for(row):
        if measure == "revenue":
            return row_metric(row, measure, rates, currency_mode)
        if measure == "quantity":
            return float(row.get("qty", 0) or 0)
        return 1.0

    if group_by != "none":
        groups = {}
        for row in filtered:
            if group_by == "month":
                key = local_date(row.get("created_at"))[:7]
            else:
                key = row.get(group_by)
            key = str(key if key is not None else "unknown")
            groups.setdefault(key, {"value": 0.0, "orders": 0, "customers": set()})
            groups[key]["value"] += value_for(row)
            groups[key]["orders"] += 1
            if row.get("customer"):
                groups[key]["customers"].add(row["customer"])
        results = []
        for key, vals in groups.items():
            item = {"group": key, "orders": vals["orders"], "unique_customers": len(vals["customers"])}
            item["value"] = round(vals["value"], 2)
            results.append(item)
        results.sort(key=lambda x: x["value"], reverse=(plan.get("sort") != "asc"))
        if plan.get("limit"):
            results = results[:int(plan["limit"])]
        unit = "USD" if measure == "revenue" and currency_mode != "original" else None
        return {"measure": measure, "group_by": group_by, "currency": unit, "results": results}

    if measure == "revenue":
        total = sum(usd_amount(r, rates) for r in filtered) if currency_mode != "original" else None
        if currency_mode == "original":
            totals = {}
            for r in filtered:
                cur = str(r.get("currency", "USD")).upper()
                totals[cur] = totals.get(cur, 0) + float(r.get("amount", 0) or 0)
            return {"revenue_by_original_currency": {k: round(v, 2) for k, v in totals.items()}, "order_count": len(filtered)}
        return {"revenue_usd": round(total, 2), "order_count": len(filtered)}
    if measure == "quantity":
        return {"quantity": round(sum(float(r.get("qty", 0) or 0) for r in filtered), 2), "order_count": len(filtered)}
    return {"count": len(filtered), "order_count": len(filtered)}

@app.on_event("startup")
def startup_load():
    try:
        rows, rates = fetch_ledger()
        app.state.rows = rows
        app.state.rates = rates
        app.state.load_error = None
    except Exception as exc:
        app.state.rows = []
        app.state.rates = USD_PER_UNIT
        app.state.load_error = str(exc)

@app.get("/")
def health():
    return {"service": "Acme Appliances ledger agent", "ready": not bool(getattr(app.state, "load_error", "not loaded")), "orders_loaded": len(getattr(app.state, "rows", []))}

@app.post("/")
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    if getattr(app.state, "load_error", None):
        raise HTTPException(status_code=503, detail="Ledger data could not be loaded. Check LEDGER_API_URL and deployment logs.")
    try:
        plan = get_plan(req.question)
        result = answer_plan(app.state.rows, app.state.rates, plan)
        return {"answer": result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not answer question: {str(exc)}")
