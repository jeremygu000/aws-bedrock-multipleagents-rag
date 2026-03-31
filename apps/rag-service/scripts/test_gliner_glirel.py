"""Quick benchmark: GLiNER + GLiREL vs Qwen-Plus on Rushing Back sample."""

from __future__ import annotations

import time

SAMPLE_TEXTS = [
    # Chunk 0: Rushing Back metadata
    (
        "Title: Rushing Back Artist: Flume feat. Vera Blue "
        "Writers: Harley Streten / Celia Pavey* / Eric Dubowsky^ / Sophie Cates+ "
        "Publishers: Kobalt Music Publishing obo Future Classic / "
        "Universal Music Publishing / Sony Music Publishing"
    ),
    # Chunk from APRA history
    (
        "In 1957, APRA signed its first television licensing agreement with GTV 9, "
        "recognising that music's future would be visual. Commercial networks followed "
        "and by the decade's end, both TV and radio were licensed. In 1965, APRA "
        "established the Silver Scroll Awards in Auckland."
    ),
    # Short organization list
    (
        "Joint statement from AMPAL, APRA AMCOS, ARIA PPCA, "
        "Australian Publishers Association, Australian Society of Authors, "
        "Australian Writers' Guild, Copyright Agency and Screenrights."
    ),
]

ENTITY_LABELS = ["Work", "Person", "Organization", "Identifier", "Territory", "LicenseTerm", "Date"]
RELATION_LABELS = [
    "WROTE",
    "PERFORMED_BY",
    "PUBLISHED_BY",
    "HAS_IDENTIFIER",
    "VALID_IN_TERRITORY",
    "HAS_TERM",
    "REFERENCES",
]


def main() -> None:
    print("=" * 80)
    print("Loading GLiNER model...")
    t0 = time.perf_counter()

    from gliner import GLiNER

    ner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    print(f"GLiNER loaded in {time.perf_counter() - t0:.1f}s")

    print("Loading GLiREL model...")
    t0 = time.perf_counter()

    from glirel import GLiREL

    rel_model = GLiREL.from_pretrained("jackboyla/glirel-large-v0")
    print(f"GLiREL loaded in {time.perf_counter() - t0:.1f}s")
    print("=" * 80)

    total_entities = 0
    total_relations = 0

    for i, text in enumerate(SAMPLE_TEXTS):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Text: {text[:120]}...")

        # NER
        t0 = time.perf_counter()
        entities = ner_model.predict_entities(text, ENTITY_LABELS, threshold=0.4)
        ner_time = time.perf_counter() - t0

        print(f"\nEntities ({len(entities)}) [{ner_time:.2f}s]:")
        for e in entities:
            print(
                f"  [{e['label']}] {e['text']} (score={e['score']:.2f}, pos={e['start']}-{e['end']})"
            )

        # RE
        ner_format = [[e["start"], e["end"], e["label"], e["text"]] for e in entities]

        t0 = time.perf_counter()
        relations = rel_model.predict_relations(
            text,
            RELATION_LABELS,
            threshold=0.3,
            ner=ner_format,
            top_k=10,
        )
        rel_time = time.perf_counter() - t0

        print(f"\nRelations ({len(relations)}) [{rel_time:.2f}s]:")
        for r in relations:
            head = r["head_text"] if isinstance(r["head_text"], str) else " ".join(r["head_text"])
            tail = r["tail_text"] if isinstance(r["tail_text"], str) else " ".join(r["tail_text"])
            print(f"  {head} --[{r['label']}]--> {tail} (score={r['score']:.2f})")

        total_entities += len(entities)
        total_relations += len(relations)

    print("\n" + "=" * 80)
    print(f"TOTALS: {total_entities} entities, {total_relations} relations")
    if total_entities > 0:
        print(f"Relation yield: {total_relations / total_entities * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
