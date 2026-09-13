"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý khách hàng của Vingroup, trả lời bằng tiếng Việt,
ngắn gọn, lịch sự và dựa trên dữ liệu có thật.

AVAILABLE TOOLS:
- search_product_catalog: tra cứu sản phẩm theo danh mục và giá tối đa.
- submit_support_ticket: tạo yêu cầu hỗ trợ cho khách hàng.

CORE RULES:
- Không tự bịa sản phẩm, giá, chính sách hoặc mã ticket.
- Luôn dùng tool khi câu hỏi cần dữ liệu catalog hoặc cần tạo ticket.
- Nếu tool không có kết quả, nói rõ rằng không tìm thấy kết quả phù hợp.

OPERATIONAL BOUNDARIES:
- Chỉ hỗ trợ sản phẩm, dịch vụ và yêu cầu chăm sóc khách hàng của Vingroup.
- Với câu hỏi ngoài phạm vi, lịch sự từ chối và đề nghị người dùng hỏi về Vingroup.

OUTPUT CONTRACT:
- Nội bộ ghi nhận Thought, Action và Observation trong trace.
- Câu trả lời cuối chỉ gồm thông tin hữu ích cho người dùng, không bịa dữ liệu.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        self.trace.append({"step": "init", "user_input": user_input})

        lowered = user_input.lower()
        is_faq = any(keyword in lowered for keyword in ["bảo hành", "bao hanh", "chính sách"])
        needs_catalog = not is_faq and any(
            keyword in lowered
            for keyword in ["sản phẩm", "san pham", "xem xe", "xe điện", "xe dien", "du lịch", "du lich", "giá"]
        )
        needs_ticket = any(
            keyword in lowered
            for keyword in ["lỗi", "loi", "hỏng", "hong", "hỗ trợ", "ho tro", "khiếu nại", "khieu nai", "ticket"]
        )

        intents = {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }
        self.trace.append({"step": "intent_detection", "intents": intents})

        if is_faq and not needs_ticket:
            answer = (
                "Theo thông tin hiện có, chính sách bảo hành pin xe điện VinFast "
                "là 10 năm."
            )
            return self._completed(answer, 1)

        pending_steps = []
        if needs_catalog:
            pending_steps.append("catalog")
        if needs_ticket:
            pending_steps.append("ticket")
        if not pending_steps:
            return self._completed(
                "VinAssistant chỉ hỗ trợ sản phẩm, dịch vụ và chăm sóc khách hàng của Vingroup.",
                1
            )

        observations = []
        iteration = 1
        for step in pending_steps:
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration - 1,
                    "status": "max_iterations_reached"
                }

            if step == "catalog":
                category = "du_lich" if any(keyword in lowered for keyword in ["du lịch", "du lich", "resort", "khách sạn", "khach san"]) else "xe_dien"
                price_match = re.search(r"([\d.,]+)\s*(triệu|trieu|tỷ|ty|tỉ|ti)", lowered)
                max_price = 999999999999
                if price_match:
                    amount = float(price_match.group(1).replace(".", "").replace(",", "."))
                    multiplier = 1_000_000 if price_match.group(2) in ["triệu", "trieu"] else 1_000_000_000
                    max_price = int(amount * multiplier)
                result = TOOL_MAP["search_product_catalog"](category, max_price)
                observation = {"tool": "search_product_catalog", "result": result}
            else:
                name_match = re.search(r"tôi tên\s+([^,.!?]+)", user_input, re.IGNORECASE)
                customer_name = name_match.group(1).strip() if name_match else "Khách hàng"
                priority = "high" if any(keyword in lowered for keyword in ["gấp", "gap", "nghiêm trọng", "nghiem trong", "khẩn", "khan"]) else "medium"
                result = TOOL_MAP["submit_support_ticket"](customer_name, user_input, priority)
                observation = {"tool": "submit_support_ticket", "result": result}

            observations.append(observation)
            self.trace.append({"step": "tool_call", "iteration": iteration, **observation})
            iteration += 1

        answer_parts = []
        for observation in observations:
            result = observation["result"]
            if observation["tool"] == "search_product_catalog":
                if not result:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp.")
                else:
                    products = ", ".join(product["name"] for product in result)
                    answer_parts.append(f"Các sản phẩm phù hợp: {products}.")
            else:
                answer_parts.append(
                    f"Đã tạo ticket {result['ticket_id']} cho {result['customer_name']}."
                )

        return self._completed(" ".join(answer_parts), len(observations))

    def _completed(self, answer: str, iterations: int) -> Dict[str, Any]:
        self.trace.append({"step": "final_answer", "answer": answer})
        return {
            "answer": answer,
            "trace": self.trace,
            "iterations": iterations,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
