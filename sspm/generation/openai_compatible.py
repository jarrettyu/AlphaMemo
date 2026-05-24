from __future__ import annotations

import json
import http.client
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .heuristic import GenerationRequest


SYSTEM_PROMPT = """You are an expert quantitative alpha researcher.
Generate exactly ONE formulaic alpha expression.

Allowed features:
$open, $close, $high, $low, $volume

Allowed operators:
CsRank(x), TsMean(x,w), TsStd(x,w), TsMax(x,w), TsMin(x,w), TsSum(x,w), TsRank(x,w),
Delay(x,d), Delta(x,d), Add(x,y), Sub(x,y), Mul(x,y), Div(x,y), Neg(x), Abs(x), Log(x),
Greater(x,y), Less(x,y), Where(cond,x,y)

Allowed windows:
5, 10, 20, 60

Rules:
- Output only the formula on one line.
- Do not use markdown.
- Keep the formula syntactically valid.
- Prefer dimensionless, rank-normalized expressions.
"""


@dataclass(slots=True)
class OpenAICompatibleGenerator:
    """OpenAI-compatible chat-completions generator.

    Works with OpenRouter by setting:
      base_url="https://openrouter.ai/api/v1"
      api_key_env="OPENROUTER_API_KEY"
    """

    model: str
    base_url: str
    api_key_env: str
    temperature: float = 0.7
    max_tokens: int = 180
    timeout: float = 60.0
    max_retries: int = 5
    retry_sleep: float = 2.0

    def generate(self, request: GenerationRequest) -> str:
        api_key = os.getenv(self.api_key_env)
        if not api_key or api_key.strip() in {"sk-or-v1-your-key-here", "sk-your-key-here", "your-key-here"}:
            raise RuntimeError(
                f"missing API key env var: {self.api_key_env}. "
                "Create alphamemo/.env from alphamemo/.env.example and set the key there, "
                f"or export {self.api_key_env} before running. "
                "The runner intentionally does not load .env.example."
            )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._build_user_prompt(request)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self._request_with_retries(payload, api_key, request)

        text = self._choice_text(data)
        formula = self._extract_formula(text)
        return formula if formula else self._fallback_formula(request)

    def _request_with_retries(self, payload: dict, api_key: str, request: GenerationRequest) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url=self.base_url.rstrip("/") + "/chat/completions",
                data=body,
                method="POST",
                headers=self._headers(api_key),
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in {408, 409, 425, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}") from exc
                last_error = RuntimeError(f"LLM request failed: HTTP {exc.code}: {detail}")
            except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.IncompleteRead) as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(self.retry_sleep * attempt)

        print(
            "[llm warning] request failed after retries; using fallback formula "
            f"for category={request.category!r}: {last_error}"
        )
        return {"choices": [{"message": {"content": self._fallback_formula(request)}}]}

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        referer = os.getenv("OPENROUTER_HTTP_REFERER")
        title = os.getenv("OPENROUTER_APP_TITLE", "AlphaMemo Alpha Mining")
        if "openrouter.ai" in self.base_url:
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title
        return headers

    def _build_user_prompt(self, request: GenerationRequest) -> str:
        parts = [
            f"Target category: {request.category}",
            f"Requested edit motif: {request.motif}",
        ]
        if request.parent_formula:
            parts.append(f"Parent factor:\n{request.parent_formula}")
            parts.append("Improve the parent while preserving a plausible financial intuition.")
        else:
            parts.append("Create a fresh factor from scratch.")
        if request.context:
            parts.append(f"Search memory context:\n{request.context}")
        return "\n\n".join(parts) + "\n\nFormula:"

    def _choice_text(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        choice = choices[0] or {}
        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        text = choice.get("text")
        return text if isinstance(text, str) else ""

    def _extract_formula(self, text: str | None) -> str:
        if not isinstance(text, str):
            return ""
        clean = text.strip()
        clean = re.sub(r"^```(?:[a-zA-Z]+)?", "", clean).strip()
        clean = clean.replace("```", "").strip()
        lines = [line.strip() for line in clean.splitlines() if line.strip()]
        for line in lines:
            line = line.strip("` ")
            if "$" in line and "(" in line:
                return line
        return lines[0].strip("` ") if lines else clean

    def _fallback_formula(self, request: GenerationRequest) -> str:
        category = (request.category or "").lower()
        if "volume" in category:
            return "CsRank(Div($volume,Add(TsMean($volume,20),1e-6)))"
        if "intraday" in category:
            return "CsRank(Div(Sub($close,$open),Add(Sub($high,$low),1e-6)))"
        if "trend" in category or "momentum" in category:
            return "CsRank(Div(Delta($close,20),Add(TsStd($close,20),1e-6)))"
        if "reversion" in category or "mean" in category:
            return "CsRank(Neg(Div(Delta($close,5),Add(TsStd($close,20),1e-6))))"
        return "CsRank(Div(Sub(TsMean($close,5),TsMean($close,20)),Add(TsStd($close,20),1e-6)))"
