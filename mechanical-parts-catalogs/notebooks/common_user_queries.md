# RAG Query Types & Indexing Strategy

---

## 5 Main User Question Types

### 1. Exact Spec Lookup
> *"Find a spur gear with 22 teeth, module 2.0, POM"*

- User knows exactly what they want
- Best handled by: **structured metadata filters**

---

### 2. Range / Constraint Query
> *"Show gears with torque ≥ 120 and diameter under 50"*

- User has numeric constraints, not a specific part in mind
- Best handled by: **numeric metadata + range filtering**

---

### 3. Part-Family / Category Exploration
> *"What spur gears do you have in module 2?"*

- User is browsing a subset of the catalog
- Best handled by: **metadata filters + part-level semantic retrieval**

---

### 4. Product Comparison / Recommendation
> *"Which gear is better for compact high-torque use?"*

- User wants a suggestion, not a lookup
- Best handled by: **embedded text + reranking + parent context**

---

### 5. Catalog Understanding / Descriptive Query
> *"What is the difference between spur gears and bevel gears?"*

- User wants conceptual explanation, not a specific part
- Best handled by: **embedded text on part/family nodes**

---

## What This Implies for Indexing

### Put in Structured Metadata
These fields should be stored as typed, filterable attributes — not embedded:

- `family`
- `module`
- `material`
- `art_nr`
- `ZZ` — teeth count
- `DM` — torque
- `ØB`, `ØTK`, `ØKK`, `ØN`, `L` — key numeric dimensions

### Put in Embedded Text
These fields should be vectorized for semantic search:

- Part title
- Short description
- Family/type wording
- Normalized semantic summary of the part
- Row text phrased naturally for semantic matching

**Example of embedded text:**
```
Spur gear, module 2.0, made of Polyacetal (POM).
This is a straight-toothed plastic spur gear from the gear catalog.
It belongs to the spur gear family and is suitable for compact power transmission applications.
Variant article number SH2022HF has 22 teeth, a width of 15 mm, a pitch circle diameter of 44 mm,
a tip diameter of 48 mm, a hub diameter of 20 mm, a length of 27 mm, and a maximum torque of 207.35 Ncm.
```

---

## Summary

| Query Type | Primary Retrieval Method |
|---|---|
| Exact spec lookup | Metadata filters |
| Range / constraint | Numeric range filters |
| Category exploration | Metadata + semantic retrieval |
| Comparison / recommendation | Semantic + reranking |
| Descriptive / conceptual | Semantic on family/part nodes |