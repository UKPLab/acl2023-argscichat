/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import { FormControl, Button, ButtonGroup, ToggleButton} from "react-bootstrap";
import { MephistoContext, CONNECTION_STATUS } from "mephisto-task";
import { AppContext, INPUT_MODE, ChatStatusBar } from "./composed/ChatApp.jsx";

import {
  CountUpTimer
} from "./index.js";


class AuthenticationForm extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            "isSending": false,
        }
        this.onMessageSend = props.onMessageSend;
    }

    trySubmit = () => {
        this.setState({
            'isSending': true,
            'pageUrl': window.location.href,
        });

        let pageUrl = window.location.href;

        this.onMessageSend({
            url: pageUrl,
            task_data: {"terminated": true}
        })
        .then(() => this.setState({
            'isSending': false,
        }));
    }

    render() {
        return (
        <div>
            <br/><br/><br/><br/><br/>
            <div className="onboarding">
               <h3><strong> Checking your authentication code...you'll be redirected to the web app in few seconds...</strong></h3>
            </div>
        </div>
        );
    }

}

export {AuthenticationForm};