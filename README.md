# SSSOmatic™
**Is your metadata suffering from semantic symptoms?**
Misalignment? Heterogeneity? Inconsistency?

**Try SSSOMatic™.**
*Discover matches. Assess relationships. Validate mappings.*

![til](sssomatic.gif)

### For symptomatic metadata.

**Indications:**
- Heterogeneous terminology
- Misaligned standards
- Ambiguous mappings
- Manual mapping fatigue

> Each dose combines lexical and semantic scoring, LLM-based candidate assessment, and human validation.

#

# Quick Setup

```bash
git clone git@github.com:CIBIO-BU/SSSOMatic.git
```

```bash
cd sssomatic
```

```bash
conda env create -f environment.yml
```

```bash
conda activate sssomatic
```

```bash
python src/sssomatic/sssomatic_workflow.py --subject-json data/FAIRe_standard.json --object-json data/mixs_terms_701.json --subject-source FAIRe --object-source MIxS --out-dir ./out/MIxS-CLI --google-api-key xxxxxxx
```