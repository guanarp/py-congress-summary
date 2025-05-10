from src.inference.semantic_splitter import SemanticSplitter

# Example transcription text (replace with your 2h transcription)
long_text = (
    "Señor Presidente, muchas gracias. Quiero comenzar recordando a todos los presentes que hoy es un día histórico para nuestro país. "
    "En segundo lugar, quiero felicitar a todos los miembros del Congreso por su dedicación y arduo trabajo durante estos últimos meses. "
    "Ahora bien, es importante destacar que aún queda mucho por hacer para garantizar el bienestar de nuestros ciudadanos. "
    "No basta con aprobar leyes; debemos asegurarnos de que sean implementadas de manera efectiva en cada rincón del país. "
    "Además, propongo que se creen comisiones especiales para abordar estos desafíos de forma integral, involucrando tanto al sector público como al privado. "
    "Permítanme también subrayar el rol de las nuevas generaciones. Los jóvenes necesitan sentirse parte de este proceso democrático. "
    "Propongo la creación de foros juveniles en cada departamento para que puedan expresar sus ideas y preocupaciones. "
    "En el ámbito económico, debemos redoblar nuestros esfuerzos para fomentar la inversión extranjera, sin descuidar nuestras industrias locales. "
    "Es vital diseñar políticas públicas que sean inclusivas y que promuevan el crecimiento sostenible. "
    "Por otro lado, no podemos ignorar el impacto del cambio climático. Propongo establecer una comisión bicameral dedicada exclusivamente al desarrollo de estrategias de mitigación y adaptación. "
    "Cada región debe ser escuchada para diseñar soluciones acordes a sus realidades particulares. "
    "Finalmente, quiero hacer un llamado a la unidad. Más allá de nuestras diferencias políticas, compartimos un mismo objetivo: construir un país más justo, equitativo y próspero. "
    "Agradezco profundamente el apoyo y la colaboración de cada uno de ustedes en este desafío histórico."
)

splitter = SemanticSplitter(max_words=64)  # For real use case I will use 4096
chunks = splitter.split_text(long_text)

for idx, chunk in enumerate(chunks):
    print(f"\n--- Chunk {idx+1} ---\n{chunk}\n")
