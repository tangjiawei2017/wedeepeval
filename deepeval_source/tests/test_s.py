from deepeval.synthesizer import Synthesizer

synthesizer = Synthesizer()


synthesizer.generate_goldens_from_docs(
    document_paths=['./document/ECIF-guimian.docx'],
    include_expected_output=True
)
print(synthesizer.synthetic_goldens)
