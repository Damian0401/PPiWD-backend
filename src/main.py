from datetime import datetime
from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from sqlalchemy import create_engine, Column, Integer, String, TIMESTAMP, ForeignKey, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import os
from flask_cors import CORS
import json
import pandas as pd
from classification.autoencoder import Autoencoder  # noqa: F401, F401, F401
from classification.classification import data_predict
from preprocessing.data_preparation import marge_raw_data_to_dataframe_with_acc_gyro
from preprocessing.data_preprocessing import extract_features

DATABASE_URL = os.getenv("DATABASE_URL")
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
        connection.execute(
            text("SELECT create_hypertable('measurement_data', by_range('timestamp'), if_not_exists => TRUE);"))
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
            },
            "/api/predict": {
                "post": {
                    "summary": "Generate prediction from device measurements",
                    "description": "Returns a single numeric prediction based on provided measurements",
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
                        "200": {
                            "description": "Prediction generated successfully",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "prediction": {"type": "number"}
                                }
                            }
                        },
                        "401": {"description": "Unauthorized"},
                        "500": {"description": "Server error"}
                    }
                }
            },
            "/api/history": {
                "get": {
                    "summary": "Fetch prediction history",
                    "description": "Returns a list of timestamped predictions within the specified date range",
                    "produces": ["application/json"],
                    "parameters": [
                        {"in": "header", "name": "X-Api-Key", "required": True, "type": "string"},
                        {"in": "query", "name": "start_date", "required": True, "type": "string", "format": "date"},
                        {"in": "query", "name": "end_date", "required": True, "type": "string", "format": "date"},
                        {"in": "mac_address", "name": "end_date", "required": True, "type": "string"}
                    ],
                    "responses": {
                        "200": {
                            "description": "History fetched successfully",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "predictions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "timestamp": {"type": "string", "format": "date-time"},
                                                "prediction": {"type": "number"}
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "401": {"description": "Unauthorized"},
                        "500": {"description": "Server error"}
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
                        "items": {"$ref": "#/definitions/MeasurementEntry"}
                    }
                }
            },
            "MeasurementEntry": {
                "type": "object",
                "required": ["type", "payload"],
                "properties": {
                    "type": {"type": "string"},
                    "payload": {"type": "array", "items": {"$ref": "#/definitions/PayloadEntry"}}
                }
            },
            "PayloadEntry": {
                "type": "object",
                "required": ["data", "timestamp"],
                "properties": {"data": {"type": "string"}, "timestamp": {"type": "integer", "format": "int64"}}
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


@app.route("/api/predict", methods=["POST"])
def predict():
    header_api_key = request.headers.get("X-Api-Key")
    if not header_api_key or header_api_key != API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401

    try:
        payload = request.get_json()
        mac_address = payload["macAddress"]
        measurements = payload["measurements"]

        session = SessionLocal()
        device_id = 1

        # Flatten and format data
        rows = []
        for measurement in measurements:
            measurement_type = get_or_create_measurement_types(session, measurement["type"])
            type_id = measurement_type.id

            for entry in measurement["payload"]:
                timestamp = datetime.fromtimestamp(entry["timestamp"] / 1000)
                data = json.dumps(entry["data"])

                rows.append({
                    "type_id": type_id,
                    "device_id": device_id,
                    "timestamp": timestamp,
                    "data": data
                })

        session.close()

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Classification logic
        df = df[df["device_id"].isin([device_id])]
        df = marge_raw_data_to_dataframe_with_acc_gyro(df)
        model_input_df = extract_features(df).iloc[[0]]

        prediction = data_predict(model_input_df)

        return jsonify({"prediction": prediction[0]['prediction']}), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to generate prediction: {str(e)}"
        }), 500



@app.route("/api/history", methods=["GET"])
def history():
    header_api_key = request.headers.get("X-Api-Key")
    if not header_api_key or header_api_key != API_KEY:
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401

    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")
    mac_address = request.args.get("mac_address")

    if not start_date_str or not end_date_str or not mac_address:
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)
    except Exception:
        return jsonify({"error": "Invalid date format. Use ISO date format YYYY-MM-DD."}), 400

    session = SessionLocal()
    try:
        device = session.query(Device).filter_by(mac_address=mac_address).first()
        if not device:
            return jsonify({"error": "Device not found"}), 404

        data_query = session.query(MeasurementData).filter(
            MeasurementData.device_id == device.id,
            MeasurementData.timestamp >= start_date,
            MeasurementData.timestamp <= end_date
        ).all()

        rows = [{
            "type_id": entry.type_id,
            "device_id": entry.device_id,
            "timestamp": entry.timestamp,
            "data": entry.data
        } for entry in data_query]

        if not rows:
            return jsonify({"predictions": []}), 200

        df = pd.DataFrame(rows)
        df = marge_raw_data_to_dataframe_with_acc_gyro(df)

        feature_df = extract_features(df)
        predictions = data_predict(feature_df)

        response = [
            {
                "timestamp": pred['timestamp'].isoformat() if isinstance(pred['timestamp'], datetime) else str(pred['timestamp']),
                "prediction": pred['prediction']
            }
            for pred in predictions
        ]

        return jsonify({"predictions": response}), 200

    except Exception as e:
        return jsonify({
            "error": f"Failed to generate prediction history: {str(e)}"
        }), 500
    finally:
        session.close()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
