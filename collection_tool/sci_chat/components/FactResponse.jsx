/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { FormControl, Button } from "react-bootstrap";
import { TextResponse } from "./index.js"
import Popup from "react-popup";
import ResizableTextArea from 'react-fluid-textarea';


/*
    Contains:
    FactResponse (1-N)
    TextResponse
*/
function FactGroup({ selection, role, onMessageSend, active, interactions, minInteractions, setHasSent }) {
    const [limitSize, setLimitSize] = React.useState(2);
    const [compSelection, setCompSelection] = React.useState("");
    const [factValues, setFactValues] = React.useState([]);
    const [compRole, setCompRole] = React.useState("");

    /* Updates internal role value -> used to ensure the correct value is used */
    React.useEffect(() => { setCompRole(role) }, [role]);

    /* Updates internal selection values -> used to ensure the correct value is used */
    React.useEffect(() => { setCompSelection(selection) }, [selection]);

    /* Updates fact values for each FactResponse component */
    React.useEffect(() => {

        let updatedFacts = [...factValues];

        /* Check if limit size has been reached */
        if (updatedFacts.length == limitSize) {
            Popup.create({
                title: null,
                content: <div> <strong> Maximum amount of supporting facts reached! Try removing a fact before adding a new one. </strong> <br/> <br/> <strong> Click anywhere to close! </strong> </div>,
                buttons: {
                    right: [{
                        text: 'Confirm',
                        className: 'success',
                        action: function() {
                            Popup.close();
                        }
                    }]
                }
            }, true);
            return
        }

        // We can add the new selection
        if (compSelection) {
            updatedFacts.push(compSelection)
            setFactValues(updatedFacts)
        }

    }, [compSelection]);

    /* Updates available facts when user removes one of them via the corresponding button */
    const tryOnRemove = React.useCallback(
        (index) => {

            // This should always be ok
            if (index < factValues.length) {
                let updatedFacts = [...factValues];
                updatedFacts.splice(index, 1);
                /*console.log("Removing fact at index -> ", index)
                console.log("Before update -> ", factValues)
                console.log("Updated facts values -> ", updatedFacts)*/
                setFactValues(updatedFacts);
            }
        }
    );

    let toRender = [];

    toRender.push(<TextResponse
    factValues={factValues}
    setFactValues={setFactValues}
    role={compRole}
    interactions={interactions}
    minInteractions={minInteractions}
    onMessageSend={onMessageSend}
    setHasSent={setHasSent}
    active={active} />);

    if (role.toLowerCase() == 'domain-expert') {
        /* General description */
        toRender.push(<div className="response-type-module"> <div>
        <h2 className="readable-header"><strong> How to add Grounding Facts </strong></h2>
        <h4><strong>Highlight with your mouse</strong> a piece of text on the given article abstract.</h4>
        <h4> You can highlight a <strong> maximum of 2 facts </strong> per reply. </h4>
        <h4><strong> Confirmed grounding facts </strong> will be displayed below. </h4>
        </div> </div>)
        /* Add components */
        for (var i = 0; i < limitSize; i++) {
            const currentFact = factValues[i]
            if (currentFact) {
                toRender.push(<FactResponse
                 factValue={currentFact}
                 listIndex={i}
                 onRemove={tryOnRemove}
                  />);
              }
        }
    }

    return toRender;
}


function FactResponse({ factValue, listIndex, onRemove }) {
    const [compFactValue, setCompFactValue] = React.useState("");
    const [compListIndex, setCompListIndex] = React.useState(0);

    /* Updates internal fact value -> used to ensure the correct value is used */
    React.useEffect(() => { setCompFactValue(factValue) }, [factValue]);

    /* Updates internal fact value -> used to ensure the correct value is used */
    React.useEffect(() => { setCompListIndex(listIndex) }, [listIndex]);

  return (
    <div className="response-type-module">
      <h4><strong> Fact {compListIndex + 1} </strong></h4>
      <div className="response-bar">
        <div className="selected-content"> {compFactValue} </div>
        <Button className="btn-cancel" onClick={(e) => onRemove(listIndex)} >
            Remove
        </Button>
      </div>
    </div>
  );
}

export {FactResponse, FactGroup};