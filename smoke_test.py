import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.chdir(r'c:\Users\Dinesh\Documents\Lexi\lexi-research-agent')

from tools.vector_search import vector_search
from tools.keyword_search import keyword_search
from tools import hybrid_search

q1 = "insurance company liability unlicensed driver motor accident"
print(f"=== Vector Search: {q1[:50]} ===")
for h in vector_search(q1, top_k=3):
    print(f"  [{h['doc_id']}] {h['case_name'][:55]} | score={h['score']}")

q2 = "Section 149 Motor Vehicles Act insurer liability"
print(f"\n=== Keyword Search: {q2[:50]} ===")
for h in keyword_search(q2, top_k=3):
    print(f"  [{h['doc_id']}] {h['case_name'][:55]} | score={h['score']}")

q3 = "unlicensed driver insurance claim denied"
print(f"\n=== Hybrid Search: {q3[:50]} ===")
for h in hybrid_search(q3, top_k=3):
    print(f"  [{h['doc_id']}] {h['case_name'][:55]} | rrf={h['rrf_score']}")

print("\nAll retrieval tools: OK")
