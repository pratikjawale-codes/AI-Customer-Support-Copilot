import time
import re
from typing import Dict, Any, List
from app.services.mock_apis import EnterpriseToolService
from app.core.rag_engine import rag_service
from app.core.llm_engine import LLMEngine
from app.core.guardrails import GuardrailsEngine
from app.core.pii_masker import PIIMasker

class AgenticWorkflowEngine:
    """
    Agentic Workflow & Tool Calling Engine
    Fulfills PDF Section 8 & 9:
    - Intent Detection
    - Dynamic Tool Selector (Order API, Payment API, Customer API, KB Search)
    - Execution Loop with Latency & Step Tracing
    - Final Response Synthesis for Human Approval
    """

    @classmethod
    def run_agent(cls, query: str, customer_id: str = "CUST-9921") -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Input Guardrail & PII Masking
        guard_res = GuardrailsEngine.validate_input(query)
        if not guard_res["is_safe"]:
            return {
                "user_query": query,
                "detected_intent": "Security Threat / Prompt Injection",
                "execution_steps": [
                    {
                        "step_number": 1,
                        "thought": "Guardrail evaluation triggered security block.",
                        "tool_name": "InputGuardrail",
                        "tool_input": {"text": query},
                        "tool_output": guard_res["block_reason"],
                        "latency_ms": round((time.time() - start_time) * 1000, 2)
                    }
                ],
                "final_response": guard_res["block_reason"],
                "requires_human_approval": False
            }

        masked_query, detected_pii = PIIMasker.mask_text(query)

        # 2. Intent Detection
        intent_res = LLMEngine.detect_intent(masked_query)
        intent = intent_res["intent"]
        entities = intent_res["extracted_entities"]
        
        order_id = entities.get("order_id", "12345")
        steps = []
        step_num = 1

        # 3. Decision Agent Reasoning & Tool Execution
        tools_context = []

        # Tool 1: KB Search (Always performed for policy alignment)
        step_start = time.time()
        kb_docs = rag_service.retrieve(masked_query, top_k=2)
        step_lat = round((time.time() - step_start) * 1000, 2)
        steps.append({
            "step_number": step_num,
            "thought": f"Searching Knowledge Base RAG for relevant policy guidelines on '{intent}'.",
            "tool_name": "search_knowledge_base",
            "tool_input": {"query": masked_query},
            "tool_output": [d["title"] for d in kb_docs],
            "latency_ms": step_lat
        })
        step_num += 1
        for d in kb_docs:
            tools_context.append(f"Policy ({d['title']}): {d['content']}")

        # Tool 2: Order / Payment API Tool Execution based on intent
        if intent in ["Delivery Delay", "Order Cancellation"] or "order" in masked_query.lower():
            step_start = time.time()
            order_info = EnterpriseToolService.get_order_status(order_id)
            step_lat = round((time.time() - step_start) * 1000, 2)
            steps.append({
                "step_number": step_num,
                "thought": f"Querying Order Enterprise API for Order ID #{order_id} tracking status.",
                "tool_name": "get_order_status",
                "tool_input": {"order_id": order_id},
                "tool_output": order_info["data"],
                "latency_ms": step_lat
            })
            step_num += 1
            tools_context.append(f"Order Data: {order_info['data']}")

        if intent in ["Payment Failure", "Refund"] or "payment" in masked_query.lower() or "deducted" in masked_query.lower():
            step_start = time.time()
            payment_info = EnterpriseToolService.check_payment_status("TXN-88219")
            step_lat = round((time.time() - step_start) * 1000, 2)
            steps.append({
                "step_number": step_num,
                "thought": "Verifying transaction capture and payment gateway refund logs.",
                "tool_name": "check_payment_status",
                "tool_input": {"transaction_id": "TXN-88219"},
                "tool_output": payment_info["data"],
                "latency_ms": step_lat
            })
            step_num += 1
            tools_context.append(f"Payment Data: {payment_info['data']}")

        # Tool 3: Customer Profile Tool
        step_start = time.time()
        cust_info = EnterpriseToolService.get_customer_details(customer_id)
        step_lat = round((time.time() - step_start) * 1000, 2)
        steps.append({
            "step_number": step_num,
            "thought": f"Fetching Customer Account & Loyalty Tier for ID {customer_id}.",
            "tool_name": "get_customer_details",
            "tool_input": {"customer_id": customer_id},
            "tool_output": cust_info["data"],
            "latency_ms": step_lat
        })
        step_num += 1

        # 4. Response Generation step
        step_start = time.time()
        generated_resp = LLMEngine.generate_responses(masked_query, tools_context)
        final_candidate = generated_resp["professional"]
        step_lat = round((time.time() - step_start) * 1000, 2)
        steps.append({
            "step_number": step_num,
            "thought": "Synthesizing retrieved tool contexts into candidate support response.",
            "tool_name": "ResponseGenerator",
            "tool_input": {"style": "professional"},
            "tool_output": "Generated response ready for Human Agent approval.",
            "latency_ms": step_lat
        })

        # Output Guardrail Check
        out_guard = GuardrailsEngine.validate_output(final_candidate)
        final_response = out_guard["sanitized_text"]

        return {
            "user_query": query,
            "detected_intent": intent,
            "execution_steps": steps,
            "final_response": final_response,
            "requires_human_approval": True
        }
