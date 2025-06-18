import streamlit as st

import sys
sys.path.append('.')

from grail_metabolism.model.grail import PretrainedGrail

model = PretrainedGrail()

st.title('Drug metabolites prediction using Grail')

input_str = st.text_input('Enter drug compound SMILES', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C')

if input_str:
    model.draw(input_str)
    with open('network.html') as file:
        st.components.v1.html(file.read())