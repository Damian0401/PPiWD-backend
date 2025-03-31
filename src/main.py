import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, ForeignKey, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError

# Database setup
DATABASE_URL = 'postgresql://postgres:postgres@timescaledb:5432/postgres'
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Retry DB connection
for i in range(10):
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✅ Connected to the database")
        break
    except OperationalError:
        print(f"⏳ Waiting for database... ({i+1}/10)")
        time.sleep(3)
else:
    raise Exception("❌ Could not connect to the database")

# Models
class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    mac_address = Column(String, unique=True, nullable=False)
    measurements = relationship("Measurement", back_populates="device")

class Measurement(Base):
    __tablename__ = "measurements"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"))
    type = Column(String, nullable=False)
    device = relationship("Device", back_populates="measurements")
    data_entries = relationship("MeasurementData", back_populates="measurement")

class MeasurementData(Base):
    __tablename__ = "measurement_data"
    measurement_id = Column(Integer, ForeignKey("measurements.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(TIMESTAMP, nullable=False, primary_key=True)
    data = Column(Text, nullable=False)
    measurement = relationship("Measurement", back_populates="data_entries")

# Drop and recreate measurement_data if timestamp column is wrong type
with engine.connect() as connection:
    result = connection.execute(text("""
        SELECT data_type
        FROM information_schema.columns
        WHERE table_name = 'measurement_data'
          AND column_name = 'timestamp';
    """)).fetchone()
    if result and result[0] != 'timestamp without time zone':
        print("⚠️ Dropping measurement_data due to wrong column type")
        connection.execute(text("DROP TABLE IF EXISTS measurement_data CASCADE;"))

# Drop and recreate measurement_data table
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS measurement_data CASCADE;"))
    conn.commit()

# Create all tables
Base.metadata.create_all(bind=engine)

# Convert to hypertable
with engine.connect() as connection:
    try:
        connection.execute(text("SELECT create_hypertable('measurement_data', 'timestamp', if_not_exists => TRUE);"))
        print("📐 Hypertable created")
    except Exception as e:
        print(f"❌ Error creating hypertable: {e}")

# Flask app
app = Flask(__name__)

# Swagger UI Setup
SWAGGER_URL = "/swagger"
API_URL = "/static/swagger.json"
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, API_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def get_or_create_device(session, mac_address):
    device = session.query(Device).filter_by(mac_address=mac_address).first()
    if not device:
        device = Device(mac_address=mac_address)
        session.add(device)
        session.commit()
    return device

def get_or_create_measurement(session, device, measurement_type):
    measurement = session.query(Measurement).filter_by(device_id=device.id, type=measurement_type).first()
    if not measurement:
        measurement = Measurement(device_id=device.id, type=measurement_type)
        session.add(measurement)
        session.commit()
    return measurement

@app.route("/save_measurements", methods=["POST"])
def save_measurements():
    data = request.get_json()
    session = SessionLocal()
    try:
        device = get_or_create_device(session, data["macAddress"])
        for measurement in data["measurements"]:
            measurement_obj = get_or_create_measurement(session, device, measurement["type"])
            for payload in measurement["payload"]:
                measurement_data = MeasurementData(
                    measurement_id=measurement_obj.id,
                    data=payload["data"],
                    timestamp=datetime.fromisoformat(payload["timestamp"])
                )
                session.add(measurement_data)
        session.commit()
        return jsonify({"message": "Measurements saved successfully"}), 201
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
