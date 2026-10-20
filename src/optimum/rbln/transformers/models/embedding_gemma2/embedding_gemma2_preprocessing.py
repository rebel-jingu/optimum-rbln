# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Request-local overlap of the pinned processor's image and audio work.

Every request decodes its own media and creates fresh processed tensors. Only
the two-worker executor persists. The owning pipeline must call ``close()``
when finished. No processor methods or global thread settings are modified.
"""

from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from math import gcd
from threading import Lock

import numpy as np
from PIL import Image

from .embedding_gemma2_profiling import trace_span


def should_parallelize(mode, audio):
    """Avoid worker overhead for less than one second of in-memory audio.

    Inspect array shapes only. File inputs stay eligible without opening them
    or probing metadata, so decoding and validation keep their original errors.
    A scalar array counts as zero here and reaches ordinary mono validation.
    """
    if mode is True:
        return True
    if mode is False:
        return False
    return not (
        all(isinstance(item, np.ndarray) for item in audio)
        and sum(item.shape[0] if item.ndim else 0 for item in audio) < 16000
    )


def _load_images(items):
    images = []
    for item in items:
        if isinstance(item, Image.Image):
            images.append(item.convert("RGB"))
        else:
            with Image.open(item) as image:
                images.append(image.convert("RGB"))
    return images


def _load_audio(items):
    waveforms = []
    for item in items:
        if isinstance(item, np.ndarray):
            waveform = item.astype(np.float32)
        else:
            import soundfile as sf
            from scipy.signal import resample_poly

            waveform, rate = sf.read(item, dtype="float32", always_2d=True)
            waveform = waveform.mean(-1)
            if rate != 16000:
                divisor = gcd(rate, 16000)
                waveform = resample_poly(waveform, 16000 // divisor, rate // divisor).astype(np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Audio must be a nonempty, finite mono waveform")
        waveforms.append(waveform)
    return waveforms


class _ProcessedMediaView:
    """Dispatch this call's media results through the unchanged processor call.

    Transformers 5.20.1's EmbeddingGemma2Processor inherits the ordinary
    ProcessorMixin.__call__. Its layout, validation, text replacement,
    tokenization and BatchFeature construction run with their original bound
    methods; only _process_images/_process_audio consume request-local results.
    """

    def __init__(self, processor, images, audio):
        self._processor, self._images, self._audio = processor, images, audio

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def _process_images(self, *args, **kwargs):
        return self._images

    def _process_audio(self, *args, **kwargs):
        return self._audio


class ParallelMediaProcessor:
    """Lazy executor owned by one pipeline, with no retained media results."""

    def __init__(self, processor):
        self.processor = processor
        self._executor = None
        self._closed = False
        self._lock = Lock()

    def _stage(self, kind, items, loaded, text, kwargs, merged):
        label = "image" if kind == "images" else "audio"
        try:
            with trace_span(f"host.prepare.{label}.decode", count=len(items)):
                values = _load_images(items) if kind == "images" else _load_audio(items)
        except BaseException as error:
            loaded.set_exception(error)
            raise
        loaded.set_result(values)
        # Use the official layout even for multiple media in a single sample.
        with trace_span(f"host.prepare.{label}.processor", count=len(items)):
            layout = self.processor.prepare_inputs_layout(text=text, **{kind: values}, **kwargs)
            if kind == "images":
                return self.processor._process_images(layout[0], **merged["images_kwargs"])
            return self.processor._process_audio(layout[3], **merged["audio_kwargs"])

    def prepare(self, *, text, images, audio, image_soft_tokens):
        kwargs = {
            "return_tensors": "pt",
            "truncation": False,
            "sampling_rate": 16000,
            "images_kwargs": {"max_soft_tokens": image_soft_tokens},
        }
        merged = self.processor._merge_kwargs(
            self.processor.valid_processor_kwargs, tokenizer_init_kwargs=self.processor.tokenizer.init_kwargs, **kwargs
        )
        image_loaded, audio_loaded = Future(), Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Parallel media processor is closed")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embeddinggemma2-media")
            # Separate contexts retain the caller's trace session on both native
            # worker lanes; a Context cannot be entered by two threads at once.
            image_job = self._executor.submit(
                copy_context().run, self._stage, "images", images, image_loaded, text, kwargs, merged
            )
            audio_job = self._executor.submit(
                copy_context().run, self._stage, "audio", audio, audio_loaded, text, kwargs, merged
            )
        try:
            # Preserve serial loader-error precedence before consuming either
            # processing result. A failing call leaves no background media work.
            loaded_images, waveforms = image_loaded.result(), audio_loaded.result()
            layout = self.processor.prepare_inputs_layout(text=text, images=loaded_images, audio=waveforms, **kwargs)
            self.processor.validate_inputs(
                images=layout[0], text=layout[1], videos=layout[2], audio=layout[3], **kwargs
            )
            image_results, audio_results = image_job.result(), audio_job.result()
            view = _ProcessedMediaView(self.processor, image_results, audio_results)
            return type(self.processor).__call__(view, text=text, images=loaded_images, audio=waveforms, **kwargs)
        finally:
            wait((image_job, audio_job))

    def close(self):
        with self._lock:
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)
