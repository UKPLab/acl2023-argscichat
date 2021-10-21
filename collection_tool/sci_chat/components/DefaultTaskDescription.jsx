/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import {Button, Collapse} from 'react-bootstrap'

function ProponentInstructions() {
    const [isOpen, setIsOpen] = React.useState(true);

    const handleClick = React.useCallback(() => {
        setIsOpen(!isOpen);
    }, [isOpen]);

    return (
      <div>
        <div className="readable-content">
        <div><h4> You can do <strong> 5 actions </strong>: </h4></div>
        <ol>
            <h4><li><strong> Start the conversation </strong> by a question about given article title</li></h4>
            <h4><li><strong> End the conversation </strong> at any time after <strong>7</strong> turns </li></h4>
            <h4><li><strong> Ask questions </strong> to know about the article,<strong> using given grounding facts </strong> so far </li></h4>
            <h4><li><strong> Express your opinion </strong> about given grounding facts so far </li></h4>
            <h4><li>Ask for <strong>DE's rebuttal </strong> </li></h4>
        </ol>
        <div><h4> For any action: </h4></div>
        <ol>
            <h4><li><strong> Actions 3-5 can be combined </strong> in a single reply </li></h4>
            <h4><li><strong>You must use your own wording and do not copy-paste content</strong></li></h4>
            <h4><li>You have maximum <strong>4 minutes to act</strong></li></h4>
        </ol>
        </div>
      </div>
    )

}

function DomainExpertInstructions() {

    return (
      <div>
        <div className="readable-content">
        <div><h4> You can do <strong> 5 actions (any combination of them) </strong>: </h4></div>
        <ol>
            <h4><li><strong> Respond to P's questions </strong></li></h4>
            <h4><li><strong> Suggest topic </strong> using grounding facts </li></h4>
            <h4><li><strong>Ask</strong> clarifying questions</li></h4>
            <h4><li><strong> Ask for P's opinion </strong> about given grounding facts so far </li></h4>
            <h4><li><strong> Support or attack P's opinion </strong> with grounding facts </li></h4>
        </ol>

        <div><h4> For any action: </h4></div>
        <ol>
            <h4><li> You can <strong>select grounding facts</strong> by highlight text with your mouse (<strong>maximum of 2 facts per reply</strong>) </li></h4>
            <h4><li><strong>You must use your own wording and do not copy-paste content</strong></li></h4>
            <h4><li>You have maximum <strong>4 minutes to act</strong></li></h4>
        </ol>
        </div>
      </div>
    )

}

function DefaultTaskDescription({ role }) {
  const [currRole, setCurrRole] = React.useState("")

  /* Updates role state */
  React.useEffect(() => {
    setCurrRole(role.toUpperCase())
   }, [role]);

  let toRender = [];

  if (currRole == "") {
    toRender.push(
    <div>
        <span>
            <h4> Role-specific instructions will be displayed here once your partner connects..</h4>
        </span>
    </div>);
  }
  else {
    if (currRole.toLowerCase() == "proponent") {
        toRender.push(
            <div>
            <ProponentInstructions/>
            </div>
        );
    }
    else {
       toRender.push(
            <div>
            <DomainExpertInstructions/>
            </div>
        );
    }
  }

  return toRender;

}

export {DefaultTaskDescription};
