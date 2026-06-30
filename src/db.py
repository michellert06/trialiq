from sqlalchemy import create_engine, Column, String, Integer, Date
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql://localhost/trialiq"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Trial(Base):
    __tablename__ = "trials"

    nct_id = Column(String, primary_key=True)
    title = Column(String)
    status = Column(String)
    phase = Column(String)
    study_type = Column(String)
    conditions = Column(String)
    lead_sponsor = Column(String)
    start_date = Column(String)
    completion_date = Column(String)

def init_db():
    Base.metadata.create_all(engine)
    print("Tables created.")

if __name__ == "__main__":
    init_db()
