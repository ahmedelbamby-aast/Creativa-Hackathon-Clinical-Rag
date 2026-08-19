#!/usr/bin/env python
"""Apply the documented source review authorized by the workspace owner.

This is intentionally a one-off, auditable data migration rather than runtime
scoring. Every answer and source locator below was checked against the two
enabled primary PDFs and their locally indexed chunks on 2026-08-19.
"""

from __future__ import annotations

import json
from pathlib import Path


PATH = Path(__file__).resolve().parent.parent / "data" / "retrieval_cases.json"
REVIEW = {"status": "reviewed", "reviewer_role": "implementation_source_review", "reviewed_at": "2026-08-19T00:00:00Z"}
IDF_DOC = "IDF_Diabetes_Atlas_11th_Edition_2025_WEB.pdf"
WHO_DOC = "WHO recommendations on care for women with diabetes during pregnancy.pdf"


def relevant(source_id: str, document: str, page: int, chunk_id: str) -> list[dict]:
    return [{"source_id": source_id, "document_name": document, "page_number": page, "chunk_id": chunk_id, "relevance_grade": 3}]


IDF_FACTS = {
    "idf_en_global_prevalence": ("589 million adults aged 20 to 79 were living with diabetes in 2024.", ["589 million", "adults aged 20 to 79", "2024"], ["589 million adults"], []),
    "idf_en_prevalence_rephrased": ("589 million adults aged 20 to 79 were living with diabetes in 2024.", ["589 million", "adults aged 20 to 79", "2024"], ["589 million adults"], []),
    "idf_ar_prevalence_rephrased": ("كان 589 مليون بالغ تتراوح أعمارهم بين 20 و79 عاماً يعيشون مع السكري في عام 2024.", ["589 مليون", "20 و79", "2024"], ["589 مليون بالغ"], []),
    "idf_ar_projection_2050": ("من المتوقع أن يرتفع العدد إلى 853 مليون بالغ بحلول عام 2050.", ["853 مليون", "2050"], ["853 مليون بالغ"], ["كم سيبلغ عدد البالغين المصابين بالسكري في عام 2050؟"]),
    "idf_en_projection_2050": ("The number is projected to rise to 853 million adults by 2050.", ["853 million", "2050"], ["853 million adults"], []),
    "idf_en_undiagnosed": ("An estimated 252 million adults living with diabetes are unaware they have the condition.", ["252 million", "unaware"], ["252 million adults"], []),
    "idf_ar_undiagnosed": ("يُقدَّر أن 252 مليون بالغ يعيشون مع السكري لا يعلمون بإصابتهم.", ["252 مليون", "لا يعلمون"], ["252 مليون بالغ"], []),
    "idf_ar_expenditure": ("أُنفق أكثر من تريليون دولار أمريكي على السكري في عام 2024.", ["أكثر من تريليون", "2024"], ["أكثر من 1 تريليون دولار"], []),
    "idf_en_expenditure_rephrased": ("More than USD 1 trillion was spent on diabetes in 2024.", ["USD 1 trillion", "2024"], ["over one trillion USD"], []),
    "idf_en_regions": ("The estimates use seven IDF Regions.", ["seven", "IDF Regions"], ["7 IDF Regions"], []),
    "idf_ar_regions": ("تُقسَّم تقديرات الاتحاد الدولي للسكري إلى سبعة أقاليم.", ["سبعة", "أقاليم"], ["7 أقاليم"], []),
    "idf_en_projection_increase": ("The projected increase is approximately 41.4%, from 589 million in 2024 to 853 million in 2050.", ["41.4", "589 million", "853 million"], ["about 41%"], []),
    "idf_ar_projection_increase": ("الزيادة المتوقعة تقارب 41.4%، من 589 مليوناً في 2024 إلى 853 مليوناً في 2050.", ["41.4", "589 مليون", "853 مليون"], ["نحو 41%"], []),
}

WHO_FACTS = {
    "who_ar_recommendation_count": ("أصدرت مجموعة تطوير الإرشادات 27 توصية.", ["27", "توصية"], ["سبع وعشرون توصية"], []),
    "who_en_recommendation_count": ("The guideline development group issued 27 recommendations.", ["27", "recommendations"], ["twenty-seven recommendations"], []),
    "who_en_monitoring_scope": ("Six WHO recommendations concern glucose monitoring during pregnancy.", ["six", "glucose monitoring"], ["6 recommendations"], []),
    "who_ar_monitoring_scope": ("تتعلق ست توصيات بمراقبة الغلوكوز أثناء الحمل.", ["ست", "مراقبة الغلوكوز"], ["6 توصيات"], []),
    "who_en_monitoring_percentage": ("Six of 27 recommendations is approximately 22.2%.", ["22.2", "six", "27"], ["about 22%"], []),
    "who_ar_monitoring_percentage": ("ست من أصل 27 توصية تساوي تقريباً 22.2%.", ["22.2", "ست", "27"], ["نحو 22%"], []),
    "who_ar_individualized_care": ("ينبغي مراعاة نهج فردي يشمل أهداف التحكم في سكر الدم وخصائص المرأة وظروفها.", ["نهج فردي", "سكر الدم"], ["النهج الفردي"], []),
    "who_en_individualized_care": ("WHO encourages an individualized approach when setting a glucose-control target during pregnancy.", ["individualized", "glucose-control"], ["individualized approach"], []),
    "who_en_gdm_lifestyle_trial": ("A two-week trial of diet and physical activity is typically conducted first.", ["two-week", "diet", "physical activity"], ["two week lifestyle trial"], []),
    "who_ar_lifestyle_trial": ("تُجرى عادة تجربة لمدة أسبوعين من النظام الغذائي والنشاط البدني أولاً.", ["أسبوعين", "النظام الغذائي", "النشاط البدني"], ["تجربة أسبوعين"], []),
    "who_ar_care_actions": ("ينبغي دعم المرأة في مراقبة سكر الدم وتقديم رعاية مناسبة لها أثناء الحمل.", ["دعم", "مراقبة سكر الدم"], ["دعم مراقبة الغلوكوز"], []),
    "who_en_care_actions": ("The woman should be supported to monitor her blood glucose during pregnancy.", ["supported", "monitor", "blood glucose"], ["support blood-glucose monitoring"], []),
}


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        case_id = case["case_id"]
        case["review"] = dict(REVIEW)
        if case_id in IDF_FACTS:
            answer, claims, aliases, variants = IDF_FACTS[case_id]
            page, chunk = (47, "IDF_Diabetes_Atlas_11th_Edition_2025_WEB_p47_c78") if case_id in {"idf_en_regions", "idf_ar_regions"} else (46, "IDF_Diabetes_Atlas_11th_Edition_2025_WEB_p46_c77")
            case.update(expected_status="ready", relevant_items=relevant("idf_diabetes_atlas_2025", IDF_DOC, page, chunk), reference_answers=[answer], required_claims=claims, accepted_aliases=aliases, query_variants=variants, task_pass_rules=["expected_status", "required_claims_present", "certified_citation_present"])
        elif case_id in WHO_FACTS:
            answer, claims, aliases, variants = WHO_FACTS[case_id]
            if "recommendation" in case_id or "monitoring" in case_id:
                page, chunk = 27, "WHO_recommendations_on_care_for_women_wi_p27_c44"
            elif "individualized" in case_id:
                page, chunk = 38, "WHO_recommendations_on_care_for_women_wi_p38_c65"
            elif "lifestyle" in case_id or "gdm" in case_id:
                page, chunk = 44, "WHO_recommendations_on_care_for_women_wi_p44_c77"
            else:
                page, chunk = 34, "WHO_recommendations_on_care_for_women_wi_p34_c57"
            case.update(expected_status="ready", relevant_items=relevant("who_diabetes_pregnancy_2025", WHO_DOC, page, chunk), reference_answers=[answer], required_claims=claims, accepted_aliases=aliases, query_variants=variants, task_pass_rules=["expected_status", "required_claims_present", "certified_citation_present"])
        else:
            vague = case_id in {"no_evidence_en_vague", "no_evidence_ar_vague"}
            case.update(expected_status="needs_clarification" if vague else "out_of_scope", relevant_items=[], reference_answers=[], required_claims=[], accepted_aliases=[], query_variants=list(case.get("query_variants", [])), task_pass_rules=["expected_status", "generation_not_called", "retrieval_not_called"] if vague else ["expected_status", "generation_not_called"])
            if case_id == "no_evidence_en_vague": case["query_variants"] = ["Can you explain?", "More details, please"]
            if case_id == "no_evidence_ar_vague": case["query_variants"] = ["اشرح أكثر", "مزيد من التفاصيل"]
    data["dataset_version"] = "2026.08-source-reviewed"
    PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
