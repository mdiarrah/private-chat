import json
from traceback import format_exc

import flask_sock
import hivemind
import torch

import config
from app import sock, models
from utils import safe_decode

logger = hivemind.get_logger(__file__)


@sock.route("/api/v2/generate")
def ws_api_generate(ws):
    try:
        request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
        assert request["type"] == "open_inference_session" #"generate"   
        #"open_inference_session"


        model_name = config.DEFAULT_MODEL_NAME #request.get("model")
        if model_name is None:
            model_name = config.DEFAULT_MODEL_NAME
        logger.info(f"ws.generate.open(), model={repr(model_name)}, max_length={repr(request['max_length'])}")

        model, tokenizer,qa = models[model_name]

        ws.send(json.dumps({"ok": True}))

        while True:
            request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
            assert request["type"] == "generate"
            inputs = request.get("inputs")
            logger.info(f"ws.generate.step(), inputs={repr(inputs)}")

            #inputs = request.get("inputs")
            #with model.inference_session(max_length=request["max_length"]) as session:
            stop = False
            if inputs is not None:
                temp0 = repr(inputs).split("###Human:")
                temp1 = ""
                UserInput = ""
                if len(temp0)> 0:
                    temp1 = temp0[len(temp0)-1].split("###")
                    if len(temp1) > 0:
                        UserInput = temp1[0].strip()
                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
                logger.info(f"ws.generate.step(), UserInput={repr(UserInput)}")
            stop_sequence = request.get("stop_sequence")
            extra_stop_sequences = request.get("extra_stop_sequences")
            while not stop:
                res = qa(UserInput)#qa(inputs)
                answer, docs = res["result"], []
                topAnswer = answer.split("\n")[0]
                combined = repr(answer)
                stop = True
                #stop = stop_sequence is None or combined.endswith(stop_sequence)
                #if extra_stop_sequences is not None:
                #    for seq in extra_stop_sequences:
                #        if combined.endswith(seq):
                #            stop = True
                if stop:
                    logger.info(f"ws.generate.step(), all_outputs={topAnswer}, stop={stop}")
                    token_count = len(combined.split())
                    ws.send(json.dumps({"ok": True, "outputs": topAnswer, "stop": stop, "token_count": token_count}))
        '''
        with model.inference_session(max_length=request["max_length"]) as session:
            ws.send(json.dumps({"ok": True}))
            inputs = request.get("inputs")
            if inputs is not None:
                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
                res = qa(inputs)
                answer, docs = res["result"], []
                stop = True
                logger.info(f"ws.generate.step(), all_outputs={repr(answer)}, stop={stop}")
                ws.send(json.dumps({"ok": True, "outputs": answer, "stop": stop, "token_count": 0}))
        '''
        
        '''
            while True:
                request = json.loads(ws.receive(timeout=config.STEP_TIMEOUT))
                assert request["type"] == "generate"
                inputs = request.get("inputs")
                logger.info(f"ws.generate.step(), inputs={repr(inputs)}")
                
                if inputs is not None:
                    inputs = tokenizer(inputs, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    n_input_tokens = inputs.shape[1]
                else:
                    n_input_tokens = 0
                

                stop_sequence = request.get("stop_sequence")
                extra_stop_sequences = request.get("extra_stop_sequences")
                #if extra_stop_sequences is not None:
                    #cont_token = tokenizer(stop_sequence, return_tensors="pt")["input_ids"].to(config.DEVICE)
                    #assert cont_token.shape == (1, 1), \
                    #    "extra_stop_sequences require stop_sequence length to be exactly 1 token"

                all_outputs = ''
                delta_q = []
                stop = False
                if not stop:
                    res = qa(inputs)
                    
                    outputs = model.generate(
                        inputs=inputs,
                        do_sample=request.get("do_sample", False),
                        temperature=request.get("temperature"),
                        top_k=request.get("top_k"),
                        top_p=request.get("top_p"),
                        repetition_penalty=request.get("repetition_penalty"),
                        max_length=request.get("max_length"),
                        max_new_tokens=request.get("max_new_tokens"),
                        session=session,
                    )
                    
                    answer, docs = res["result"], []
                    #delta = answer[0, n_input_tokens:].tolist()
                    #outputs = safe_decode(tokenizer, delta_q + delta)
                    inputs = None  # Inputs are passed only for the 1st token of the bot's response
                    n_input_tokens = 0
                    combined = all_outputs + answer #outputs
                    #stop = stop_sequence is None or combined.endswith(stop_sequence)
                    stop = True
                    
                    if extra_stop_sequences is not None:
                        for seq in extra_stop_sequences:
                            if combined.endswith(seq):
                                stop = True
                                session.last_token_id = cont_token
                    
                    if not stop:
                        # If there's a replacement character, keep getting more tokens
                        # until we can decode properly
                        #delta_q = delta_q + delta
                        logger.info(f"ws.generate.append_retry(), all_outputs={repr(combined)}")
                    else:
                        all_outputs = combined
                        #token_count = len(delta_q + delta)
                        #delta_q = []
                        logger.info(f"ws.generate.step(), all_outputs={repr(all_outputs)}, stop={stop}")
                        ws.send(json.dumps({"ok": True, "outputs": answer, "stop": stop, "token_count": 0}))
                        '''
    except flask_sock.ConnectionClosed:
        pass
    except Exception:
        logger.warning("ws.generate failed:", exc_info=True)
        ws.send(json.dumps({"ok": False, "traceback": format_exc()}))
    finally:
        logger.info(f"ws.generate.close()")
