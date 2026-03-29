import streamlit as st

from grail_metabolism.model.grail import PretrainedGrail


@st.cache_resource
def load_model() -> PretrainedGrail:
    return PretrainedGrail(strict=False)


st.title("GRAIL metabolite prediction")
smiles = st.text_input("Enter substrate SMILES", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C")

if smiles:
    model = load_model()
    try:
        html_path = model.draw(smiles, output_html="network.html")
        with open(html_path) as handle:
            st.components.v1.html(handle.read(), height=850)
    except Exception as exc:
        st.error(str(exc))
