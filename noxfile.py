import nox

@nox.session(python=["3.14", "3.13", "3.12", "3.11", "3.10", "3.9"])
def tests(session):
    session.install(".") # Instala sirena en el entorno temporal
    session.run("sirena", "inputs/fermion_test.txt", f"outputs/output_{session.python}.txt", silent=False)

