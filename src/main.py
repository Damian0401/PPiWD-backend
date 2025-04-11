import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, ForeignKey, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError
import os
from flask_cors import CORS
import json

DATABASE_URL = os.getenv("DATABASE_URL")
API_HOST = os.getenv("API_HOST")
API_KEY = os.getenv("API_KEY")

# Database setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Models
class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    mac_address = Column(String, unique=True, nullable=False)
    measurements = relationship("MeasurementData", back_populates="device")

class MeasurementType(Base):
    __tablename__ = "measurement_types"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)
    data_entries = relationship("MeasurementData", back_populates="measurement")

class MeasurementData(Base):
    __tablename__ = "measurement_data"
    type_id = Column(Integer, ForeignKey("measurement_types.id", ondelete="CASCADE"), primary_key=True)
    device_id = Column(Integer, ForeignKey("devices.id", ondelete="CASCADE"), primary_key=True)
    timestamp = Column(TIMESTAMP, nullable=False, primary_key=True)
    data = Column(Text, nullable=False)
    device = relationship("Device", back_populates="measurements")
    measurement = relationship("MeasurementType", back_populates="data_entries")

# Create all tables
Base.metadata.create_all(bind=engine)

# Convert to hypertable
with engine.connect() as connection:
    try:
        connection.execute(text("SELECT create_hypertable('measurement_data', by_range('timestamp'), if_not_exists => TRUE);"))
        print("Hypertable created")
    except Exception as e:
        print(f"Error creating hypertable: {e}")

# Flask app
app = Flask(__name__)

CORS(app)

# Swagger UI Setup
SWAGGER_URL = "/swagger"
OPENAPI_URL = "/swagger.json"
swaggerui_blueprint = get_swaggerui_blueprint(SWAGGER_URL, OPENAPI_URL)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

def get_or_create_device(session, mac_address):
    device = session.query(Device).filter_by(mac_address=mac_address).first()
    if not device:
        device = Device(mac_address=mac_address)
        session.add(device)
        session.commit()
    return device

def get_or_create_measurement_types(session, measurement_type):
    measurement = session.query(MeasurementType).filter_by(type=measurement_type).first()
    if not measurement:
        measurement = MeasurementType(type=measurement_type)
        session.add(measurement)
        session.commit()
    return measurement

@app.route("/swagger.json")
def swagger_spec():
    return jsonify({
        "swagger": "2.0",
        "info": {
            "title": "Measurement API",
            "version": "1.0"
        },
        "host": API_HOST,
        "basePath": "/",
        "paths": {
            "/api/measurements": {
                "post": {
                    "summary": "Save device measurements",
                    "description": "Saves measurements from a device",
                    "consumes": ["application/json"],
                    "produces": ["application/json"],
                    "parameters": [
                        {
                            "in": "header",
                            "name": "X-Api-Key",
                            "required": True,
                            "type": "string"
                        },
                        {
                            "in": "body",
                            "name": "body",
                            "required": True,
                            "schema": {
                                "$ref": "#/definitions/MeasurementPayload"
                            }
                        }
                    ],
                    "responses": {
                        "201": {
                            "description": "Measurements saved successfully",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"}
                                }
                            }
                        },
                        "500": {
                            "description": "Server error",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "error": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "definitions": {
            "MeasurementPayload": {
                "type": "object",
                "required": ["macAddress", "measurements"],
                "properties": {
                    "macAddress": {"type": "string"},
                    "measurements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["type", "payload"],
                            "properties": {
                                "type": {"type": "string"},
                                "payload": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "required": ["data", "timestamp"],
                                        "properties": {
                                            "data": {"type": "string"},
                                            "timestamp": {
                                            "type": "integer",
                                            "format": "int64",
                                            "description": "Timestamp in milliseconds since epoch"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })



@app.route("/api/measurements", methods=["POST"])
def save_measurements():
    header_api_key = request.headers.get("X-Api-Key")
    if not header_api_key or header_api_key != API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401

    data = request.get_json()
    session = SessionLocal()
    try:
        device = get_or_create_device(session, data["macAddress"])
        for measurement in data["measurements"]:
            measurement_obj = get_or_create_measurement_types(session, measurement["type"])
            for payload in measurement["payload"]:
                measurement_data = MeasurementData(
                    type_id=measurement_obj.id,
                    device_id=device.id,
                    data=json.dumps(payload["data"]),
                    timestamp=datetime.fromtimestamp(payload["timestamp"] / 1000)
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
