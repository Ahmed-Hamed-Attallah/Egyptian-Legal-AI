"""
Prompt templates for Egyptian legal question answering.
"""
from __future__ import annotations

from typing import Dict, List


class LegalPromptTemplates:
    SYSTEM_PROMPT_ARABIC = """أنت مساعد قانوني متخصص في القانون المصري. مهمتك هي الإجابة على الأسئلة القانونية بناءً على النصوص القانونية المقدمة لك فقط.

القواعد الأساسية:
أجب فقط بناءً على النصوص القانونية المرفقة في السياق.
لا تختر أرقام مواد أو قوانين أو اقتباسات قانونية.
إذا لم تكن المعلومات المقدمة كافية للإجابة، قل بوضوح: "لا تتوفر معلومات كافية في المصادر المقدمة للإجابة على هذا السؤال."
ميّز بين النص القانوني الأصلي وتفسيرك أو شرحك.
انتبه لتواريخ النفاذ والتعديلات - لا تقدم نسخاً ملغاة من القوانين كأنها سارية.
استخدم الاستشهادات القانونية بدقة عند الإشارة إلى مواد محددة.
إذا كان السؤال خارج نطاق النصوص المقدمة، اذكر ذلك بوضوح.

يجب أن تكون إجاباتك:
دقيقة قانونياً
مدعومة بالنصوص المقدمة
واضحة ومباشرة
باللغة العربية الفصحى"""

    SYSTEM_PROMPT_ENGLISH = """You are a legal assistant specialized in Egyptian law. Your task is to answer legal questions based ONLY on the legal texts provided to you.

Core rules:
Answer ONLY based on the legal texts provided in the context.
Do NOT invent article numbers, law numbers, or legal quotations.
If the provided information is insufficient to answer, clearly state: "Insufficient information in the provided sources to answer this question."
Distinguish between original legal text and your interpretation or explanation.
Pay attention to effective dates and amendments - do not present repealed versions as currently effective.
Use legal citations accurately when referencing specific articles.
If the question is outside the scope of the provided texts, state this clearly.

Your answers must be:
Legally accurate
Supported by provided texts
Clear and direct
In proper Arabic (for legal content)"""

    # ------------------------------------------------------------------
    # Conversation memory templates
    # ------------------------------------------------------------------
    MEMORY_DIGEST_TEMPLATE = """## سجل المحادثات السابقة للمستخدم (للمرجعية فقط)
{entries}

قواعد التعامل مع سجل المحادثات:
- إذا أشار المستخدم إلى سؤال سابق أو محادثة سابقة، استخدم السجل أعلاه فقط للإجابة.
- اذكر اسم المحادثة السابقة عند الإشارة إليها إن وُجد.
- لا تخترع أسئلة أو إجابات غير موجودة في السجل.
- سجل المحادثات لا يُعد مصدراً قانونياً؛ المصادر القانونية هي النصوص المرفقة في السياق فقط."""

    MEMORY_DIGEST_ENTRY_TEMPLATE = (
        '- محادثة "{title}" (أنشئت في {created_at}):\n'
        "  الأسئلة السابقة: {questions}\n"
        "  آخر إجابة: {answer_snippet}"
    )

    USER_PROMPT_TEMPLATE = """السؤال: {question}

السياق القانوني:
{context}

بناءً على النصوص القانونية أعلاه، أجب على السؤال بدقة. إذا لم تكن المعلومات كافية، اذكر ذلك بوضوح."""

    USER_PROMPT_TEMPLATE_EN = """Question: {question}

Legal Context:
{context}

Based on the legal texts above, answer the question accurately. If the information is insufficient, state this clearly."""

    # ------------------------------------------------------------------
    # Existing helpers (unchanged)
    # ------------------------------------------------------------------
    @classmethod
    def get_system_prompt(cls, language: str = "ar") -> str:
        """Get system prompt in specified language."""
        if language.lower() == "en":
            return cls.SYSTEM_PROMPT_ENGLISH
        return cls.SYSTEM_PROMPT_ARABIC

    @classmethod
    def format_user_prompt(cls, question: str, context: str, language: str = "ar") -> str:
        """Format user prompt with question and context."""
        if language.lower() == "en":
            return cls.USER_PROMPT_TEMPLATE_EN.format(question=question, context=context)
        return cls.USER_PROMPT_TEMPLATE.format(question=question, context=context)

    # ------------------------------------------------------------------
    # Conversation memory helpers (new)
    # ------------------------------------------------------------------
    @classmethod
    def format_memory_digest(cls, digest: List[Dict[str, object]]) -> str:
        """Render the cross-chat digest as an Arabic reference block.

        Returns an empty string when there is nothing worth including,
        so callers can append it unconditionally.
        """
        if not digest:
            return ""
        entries: List[str] = []
        for item in digest:
            questions = item.get("questions") or []
            if not questions:
                continue  # a conversation without questions adds no value
            entries.append(
                cls.MEMORY_DIGEST_ENTRY_TEMPLATE.format(
                    title=item.get("title") or "محادثة بدون عنوان",
                    created_at=item.get("created_at") or "تاريخ غير معروف",
                    questions=" | ".join(str(q) for q in questions),
                    answer_snippet=item.get("last_answer_snippet") or "غير متوفرة",
                )
            )
        if not entries:
            return ""
        return cls.MEMORY_DIGEST_TEMPLATE.format(entries="\n".join(entries))

    @classmethod
    def build_system_prompt_with_memory(
        cls,
        memory_digest: List[Dict[str, object]],
        language: str = "ar",
    ) -> str:
        """System prompt + cross-chat memory block (if any).

        Note: current-conversation history is NOT rendered here; it is passed
        as real message turns to the provider (see AnswerGenerator).
        """
        system_prompt = cls.get_system_prompt(language)
        memory_block = cls.format_memory_digest(memory_digest or [])
        if memory_block:
            return f"{system_prompt}\n\n{memory_block}"
        return system_prompt