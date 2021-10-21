/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import DoneButton from "./DoneButton.jsx";
import Popup from "react-popup";

function DoneResponse({ onTaskComplete, onMessageSend, isTaskDone, doneText, agentStatus }) {
    const [showPopup, setShowPopup] = React.useState(true);

    // Run just once
    React.useEffect(() => {
        console.log('agentStatus -> ', agentStatus);
        if (isTaskDone && showPopup && agentStatus != 'partner disconnect' && agentStatus != 'timeout' && agentStatus != 'disconnect') {
            setTimeout(function() {Popup.close(); onTaskComplete()}, 5000);
            Popup.create({
                title: "Dialogue Completed!",
                content: <div> {doneText}! <br/> The dialogue will be automatically submitted in 5 seconds... <br/><br/> <strong> PLEASE WAIT </strong> </div>,
                buttons: {
                    right: [{
                        text: 'Dismiss',
                        className: 'success',
                        action: function() {
                            Popup.close();
                            onTaskComplete();
                        }
                    }]
                }
            }, true);

            setShowPopup(false);
        }

        if ((agentStatus == 'timeout' || agentStatus == 'partner disconnect' || agentStatus == 'disconnect') && showPopup) {
            setTimeout(function() {Popup.close(); onTaskComplete()}, 5000);
            Popup.create({
                title: "Something went wrong!",
                content: <div> {doneText}! <br/> The dialogue will be automatically submitted in 5 seconds... <br/><br/> <strong> PLEASE WAIT </strong> </div>,
                buttons: {
                    right: [{
                        text: 'Dismiss',
                        className: 'success',
                        action: function() {
                            Popup.close();
                            onTaskComplete();
                        }
                    }]
                }
            }, true);

            setShowPopup(false);
        }
    }, [agentStatus, isTaskDone]);

  // TODO maybe move to CSS?
  let pane_style = {
    paddingLeft: "25px",
    paddingTop: "20px",
    paddingBottom: "20px",
    paddingRight: "25px",
    float: "left",
  };
  return (
    <div
      id="response-type-done"
      className="response-type-module"
      style={pane_style}
    >
      <h4><strong>Your conversation has ended! What's next?</strong></h4> <br/>
      <ol>
        <h4><li> Please, wait until automatic submission has completed (a popup should appear at the end) </li></h4>
        <h4><li> You can close this tab (please, make sure that the status bar displays 'disconnected' -- scroll up) </li></h4>
        <h4><li> Check your remaining dialogue sessions (<strong>if any</strong>) in the <a href="http://137.204.57.12/sessions.html"> Session Calendar </a> section. </li></h4>
        <h4><li> Join your next dialogue sessions (<strong>when it's time!</strong>): Please, check <a href="http://137.204.57.12/application.html">Web Application</a> section. </li></h4>
      </ol>
    </div>
  );
}

export default DoneResponse;
