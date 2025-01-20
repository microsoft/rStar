# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import os
import time
from tqdm import tqdm
import concurrent.futures
from openai import AzureOpenAI

client = AzureOpenAI(
    api_version="",
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
)

max_threads = 32


def load_OpenAI_model(model):
    return None, model


def generate_with_OpenAI_model(
    prompt,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
):
    messages = [{"role": "user", "content": prompt}]
    parameters = {
        "model": model_ckpt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stop": stop,
        "seed": 1,
    }

    ans, timeout = "", 5
    while not ans:
        try:
            time.sleep(timeout)
            completion = client.chat.completions.create(messages=messages, **parameters)
            ans = completion.choices[0].message.content

        except Exception as e:
            print(e)
        if not ans:
            timeout = timeout * 2
            if timeout > 120:
                timeout = 1
            try:
                print(f"Will retry after {timeout} seconds ...")
            except:
                pass
    return ans


def generate_n_with_OpenAI_model(
    prompt,
    n=1,
    model_ckpt="gpt-35-turbo",
    max_tokens=256,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    stop=["\n"],
    max_threads=3,
    disable_tqdm=True,
):
    preds = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(generate_with_OpenAI_model, prompt, model_ckpt, max_tokens, temperature, top_k, top_p, stop)
            for _ in range(n)
        ]
        for i, future in tqdm(
            enumerate(concurrent.futures.as_completed(futures)),
            total=len(futures),
            desc="running evaluate",
            disable=disable_tqdm,
        ):
            ans = future.result()
            preds.append(ans)
    return preds
