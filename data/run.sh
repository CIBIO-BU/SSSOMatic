conda env create -f environment.yml

conda activate sssomatic

pip install -e .

sssomatic --subject-json data/FAIRe_YAML_slots/faire_yaml_slot.json --subject-source FAIRe --object-json data/DwC_DP/dwc_dp_fields.json --object-source DwC-DP --out-dir out/DwC