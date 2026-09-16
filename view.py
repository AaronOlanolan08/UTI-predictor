from pathlib import Path
import pickle
import re
import os
from tempfile import NamedTemporaryFile
from uuid import uuid4

import pandas as pd
from flask import Flask, flash, jsonify, render_template, request
from pypdf import PdfReader
from rapidocr_onnxruntime import RapidOCR
import pymupdf
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "baseline_model16.pkl"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "local-development-secret")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}

with MODEL_PATH.open("rb") as model_file:
	model = pickle.load(model_file)

ocr_engine = RapidOCR()


OPTIONS = {
	"gender": ["FEMALE", "MALE"],
	"color": [
		"AMBER", "BROWN", "DARK YELLOW", "LIGHT RED", "LIGHT YELLOW",
		"RED", "REDDISH", "REDDISH YELLOW", "STRAW", "YELLOW",
	],
	"transparency": ["CLEAR", "CLOUDY", "HAZY", "SLIGHTLY HAZY", "TURBID"],
	"glucose": ["NEGATIVE", "TRACE", "1+", "2+", "3+", "4+"],
	"protein": ["NEGATIVE", "TRACE", "1+", "2+", "3+"],
	"epithelial_cells": ["NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "PLENTY", "LOADED"],
	"mucous_threads": ["NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "PLENTY"],
	"bacteria": ["RARE", "FEW", "OCCASIONAL", "MODERATE", "PLENTY", "LOADED"],
	"amorphous_urates": ["NONE SEEN", "RARE", "FEW", "OCCASIONAL", "MODERATE", "PLENTY"],
}


def save_upload(upload):
	if not upload or not upload.filename:
		return None
	suffix = Path(secure_filename(upload.filename)).suffix.lower()
	if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
		return None
	filename = f"{uuid4().hex}{suffix}"
	upload.save(UPLOAD_DIR / filename)
	return filename


def read_upload_text(file_path):
	if file_path.suffix.lower() == ".pdf":
		text = "\n".join(page.extract_text() or "" for page in PdfReader(file_path).pages)
		if text.strip():
			return text

		pdf = pymupdf.open(file_path)
		ocr_lines = []
		for page in pdf:
			pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
			with NamedTemporaryFile(suffix=".png") as image_file:
				image_file.write(pixmap.tobytes("png"))
				image_file.flush()
				ocr_result, _ = ocr_engine(image_file.name)
				ocr_lines.extend(item[1] for item in ocr_result or [])
		return "\n".join(ocr_lines)

	ocr_result, _ = ocr_engine(str(file_path))
	return "\n".join(item[1] for item in ocr_result or [])


def extract_fields(file_path):
	text = read_upload_text(file_path).upper()
	compact = re.sub(r"[^A-Z0-9+.-]", "", text)
	fields = {}

	def section(label, following_labels):
		compact_label = re.sub(r"[^A-Z0-9]", "", label.upper())
		start = compact.find(compact_label)
		if start < 0:
			return ""
		end = len(compact)
		for following_label in following_labels:
			next_start = compact.find(re.sub(r"[^A-Z0-9]", "", following_label.upper()), start + 1)
			if next_start >= 0:
				end = min(end, next_start)
		return compact[start + len(compact_label):end]

	def option_field(name, label, following_labels):
		value_section = section(label, following_labels)
		for option in OPTIONS[name]:
			if re.sub(r"[^A-Z0-9]", "", option) in value_section:
				fields[name] = option
				return

	option_field("color", "COLOR", ["TRANSPARENCY"])
	option_field("transparency", "TRANSPARENCY", ["VOLUME", "WBC"])
	option_field("protein", "PROTEIN", ["SUGAR", "BACTERIA"])
	option_field("glucose", "SUGAR", ["WBC", "EPITHELIAL CELLS"])
	option_field("epithelial_cells", "EPITHELIAL CELLS", ["AMORPHOUS URATES"])
	option_field("amorphous_urates", "AMORPHOUS URATES", ["MUCOUS THREADS"])
	option_field("mucous_threads", "MUCOUS THREADS", ["BACTERIA"])
	option_field("bacteria", "BACTERIA", [])

	for name, label, following_labels in [
		("specific_gravity", "SPECIFIC GRAVITY", ["PH REACTION"]),
		("ph", "PH REACTION", ["PROTEIN"]),
		("wbc", "WBC", ["RBC", "EPITHELIAL CELLS"]),
		("rbc", "RBC", ["EPITHELIAL CELLS"]),
	]:
		value_section = section(label, following_labels)
		range_match = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)", value_section)
		if range_match:
			fields[name] = str(round((float(range_match.group(1)) + float(range_match.group(2))) / 2, 3))
		else:
			number_match = re.search(r"\d+(?:\.\d+)?", value_section)
			if number_match:
				fields[name] = number_match.group(0)

	age_match = re.search(r"AGE/SEX\s*:?\s*(\d+)", text)
	if age_match:
		fields["age"] = age_match.group(1)

	return fields


@app.route("/", methods=["GET", "POST"])
def index():
	result = None
	form_values = request.form.to_dict() if request.method == "POST" else {}

	if request.method == "POST":
		try:
			upload = request.files.get("sample_file")
			if upload and upload.filename:
				filename = save_upload(upload)
				form_values.update(extract_fields(UPLOAD_DIR / filename))
			features = pd.DataFrame([{
				"WBC": float(form_values["wbc"]),
				"Transparency": form_values["transparency"],
				"Epithelial Cells": form_values["epithelial_cells"],
				"Bacteria": form_values["bacteria"],
			}])
			prediction = int(model.predict(features)[0])
			probabilities = model.predict_proba(features)[0]
			result = {
				"label": "POSITIVE" if prediction == 1 else "NEGATIVE",
				"confidence": round(float(probabilities[prediction]) * 100, 1),
			}
		except (KeyError, TypeError, ValueError) as error:
			flash(f"Please check the form values: {error}", "error")
		except Exception:
			flash("The sample could not be analyzed. Please try again.", "error")

	return render_template("index.html", options=OPTIONS, result=result, form_values=form_values)


@app.post("/extract")
def extract():
	upload = request.files.get("sample_file")
	if not upload or not upload.filename:
		return jsonify({"error": "Choose a PDF or image first."}), 400
	filename = save_upload(upload)
	if not filename:
		return jsonify({"error": "Only PDF, JPG, JPEG, and PNG files are supported."}), 400
	try:
		return jsonify({"fields": extract_fields(UPLOAD_DIR / filename)})
	except Exception as error:
		return jsonify({"error": f"Could not read this file: {error}"}), 422


if __name__ == "__main__":
	app.run(
		debug=os.environ.get("FLASK_DEBUG", "0") == "1",
		host="0.0.0.0",
		port=int(os.environ.get("PORT", "5001")),
	)
