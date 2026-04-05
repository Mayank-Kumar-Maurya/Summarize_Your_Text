from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Initialize our fastapi app
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model & tokenizer
model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".")

# Input Schema for dialogue => string
class DialogueInput(BaseModel):
  dialogue: str



def clean_data(text):
  # remove lines
  text = re.sub(r"\r\n", " ", text)
  # remove extra spaces
  text = re.sub(r"\s+", " ", text)
  # remove html tags
  text = re.sub(r"<.*?>", " ", text)
  # remove starting ending extra spaces & convert to lower-case
  text = text.strip().lower()

  return text



def summarize_dialogue(dialogue : str) -> str:
  dialogue = clean_data(dialogue) ## clean

  # tokenize
  inputs = tokenizer(
      dialogue,
      padding="max_length",
      max_length=512,
      truncation=True,
      return_tensors="pt"  ## this will return pytorch tensors
  ).to(device)

  # generate the text(summary) => which is in the form of token ids
  model.to(device)
  targets = model.generate(
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      max_length=150,
      num_beams=4,   ## in Ai we have an algorithm called beam search, num_beams=4 means our transformer will going to generate 4 sequences of output and finally give that summary which is best out of them
      early_stopping=True   ## jaise hi hume end of sequence mil jai wahi hume stop karjana hai
  )

  # token ids convert to text(summary) => decoding
  summary = tokenizer.decode(targets[0], skip_special_tokens=True) ## skip_special_token means skip the EOS(end of sequ.), seperators, tab spaces, tags etc
  return summary


# API Endpoints
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
  summary = summarize_dialogue(dialogue_input.dialogue)
  return {"summary": summary}


@app.post("/", response_class=HTMLResponse)
async def home(request: Request):
  return templates.TemplateResponse("index.html", {"request": request})