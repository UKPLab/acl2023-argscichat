/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { FormControl, Button, ButtonGroup, ToggleButton } from "react-bootstrap";
import Popup from "react-popup";

function TextResponse({ factValues, setFactValues, role, onMessageSend, active, interactions, minInteractions, setHasSent }) {
  const [textValue, setTextValue] = React.useState("");
  const [sending, setSending] = React.useState(false);
  const [compFactValues, setCompFactValues] = React.useState(["", ""]);
  const [additionalData, setAdditionalData] = React.useState("");
  const [terminated, setTerminated] = React.useState(false);
  const inputRef = React.useRef();
  const [compInteractions, setCompInteractions] = React.useState(0);
  const [compMinInteractions, setCompMinInteractions] = React.useState(0);

  /* Makes sure to use updated values of factValues prop */
  React.useEffect(() => { setCompFactValues(factValues) }, [factValues]);

  /* Transforms factValues prop into text format */
  React.useEffect(() => {
    let factData = ""
    let counter = 0
    for (var i = 0; i < 2; i++) {
        if (compFactValues[i]) {
            factData += (counter+1) + ". " + compFactValues[i] + "\n"
            counter += 1
        }
    }
    setAdditionalData(factData);
  }, [compFactValues]);

  /* Update interaction values for Terminate Dialogue checkbox */
  React.useEffect(() => {
    setCompInteractions(interactions);
  }, [interactions]);

  React.useEffect(() => {
    setCompMinInteractions(minInteractions);
  }, [minInteractions]);

  React.useEffect(() => {
    if (active && inputRef.current && inputRef.current.focus) {
      inputRef.current.focus();
    }
  }, [active]);

  const executeMessageSend = React.useCallback(() => {
      setSending(true);
      setHasSent(true);
      let completeMessage = textValue

      if (role.toLowerCase() == 'domain-expert' && additionalData) {
        completeMessage += "\n Facts: \n" + additionalData
      }

      onMessageSend({ text: completeMessage, task_data: {"terminated": terminated} }).then(() => {
        setTextValue("");
        setFactValues([]);
        setSending(false);
        setHasSent(false);
      });
  });

  const tryMessageSend = React.useCallback(() => {
    if (textValue !== "" && active && !sending) {

        // Check if some facts have been selected before sending the reply
        if (role.toLowerCase() == 'domain-expert' && factValues.length == 0) {

            Popup.create({
                title: "Warning! No facts selected!",
                content: <div> No supporting facts have been selected! Are you sure? Please confirm to proceed. <br/> <br/> <strong> Click anywhere to close! </strong> </div>,
                buttons: {
                    left: [{
                        text: 'Close',
                        className: 'danger',
                        action: function () {
                            Popup.close();
                        }
                            }],
                    right: [{
                        text: 'Confirm',
                        className: 'success',
                        action: function() {
                            executeMessageSend();
                            Popup.close();
                        }
                    }]
                }
            }, true);
            return
        }

      executeMessageSend();
    }
  }, [textValue, active, sending, onMessageSend, compFactValues, additionalData, terminated]);

  const handleKeyPress = React.useCallback(
    (e) => {
      if (e.key === "Enter") {
        tryMessageSend();
        e.stopPropagation();
        e.nativeEvent.stopImmediatePropagation();
      }
    },
    [tryMessageSend]
  );

  let toRender = [];

  toRender.push(<div className="response-type-module">
      <div className="response-bar">
        <FormControl
          type="text"
          spellCheck="true"
          id="id_send_msg_text"
          className="response-text-input"
          inputRef={(ref) => {
            inputRef.current = ref;
          }}
          value={textValue}
          placeholder="Please enter here..."
          onKeyPress={(e) => handleKeyPress(e)}
          onChange={(e) => setTextValue(e.target.value)}
          disabled={!active || sending}
        />
        <Button
          className="btn btn-primary submit-response"
          id="id_send_msg_button"
          disabled={textValue === "" || !active || sending}
          onClick={() => tryMessageSend()}
        >
          Send
        </Button>
        </div>

        <div className="response-type-module">
          <br/>
          <h4><strong><u>NOTE:</u> DO NOT write TOO LONG messages, but prefer splitting the conversation into multiple turns. </strong></h4>
          <h4><strong><u> NOTE: </u> Remember that your knowledge is limited to available content! </strong></h4>
        </div>

    </div>);

    if (role.toLowerCase() == 'proponent') {
        toRender.push(
        <div className="response-type-module">
        <h4> Mark to terminate conversation with your next message (<strong> Available after {compMinInteractions} interactions </strong>): </h4>
            <ToggleButton
                type="checkbox"
                variant="secondary"
                checked={terminated}
                disabled={compMinInteractions == 0 || compInteractions < compMinInteractions}
                value="1"
                onChange={(e) => setTerminated(e.currentTarget.checked)}
            >
            <h4>Terminate Dialogue</h4>
            </ToggleButton>
        </div>
        );
    }

  return toRender;
}


export default TextResponse;