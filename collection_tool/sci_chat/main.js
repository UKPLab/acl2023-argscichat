/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import "./style.css";
import { Passage, logSelection } from "./components/core_components.jsx";
import { ChatApp } from "./components/composed/ChatApp.jsx";
import { DefaultTaskDescription } from "./components/DefaultTaskDescription.jsx"
import { ChatMessage } from "bootstrap-chat";
import Popup from "react-popup";

function RenderChatMessage({ message, mephistoContext, appContext, idx }) {
  const { agentId } = mephistoContext;
  let { currentAgentNames } = appContext.taskContext;

  if (!currentAgentNames) {
    currentAgentNames = [];
  }

  return (
    <div>
      <ChatMessage
        isSelf={message.id === agentId || message.id in currentAgentNames}
        agentName={
          message.id in currentAgentNames
            ? currentAgentNames[message.id]
            : message.id
        }
        message={message.text}
        taskData={message.task_data}
        messageId={message.message_id}
      />
    </div>
  );
}

function MainApp() {
  const [abstract, setAbstract] = React.useState("");
  const [title, setTitle] = React.useState("");
  const [introduction, setIntroduction] = React.useState("");
  const [selection, setSelection] = React.useState("");
  const [role, setRole] = React.useState("");
  const [interactions, setInteractions] = React.useState(0);
  const [minInteractions, setMinInteractions] = React.useState(0);

  return (
    <ChatApp
      renderMessage={({ message, idx, mephistoContext, appContext }) => (
        <RenderChatMessage
          message={message}
          mephistoContext={mephistoContext}
          appContext={appContext}
          idx={idx}
          key={message.message_id + "-" + idx}
        />
      )}
      renderRightPane={({ mephistoContext: { taskConfig } }) => (
         <Passage
            abstract={abstract}
            title={title}
            introduction={introduction}
            onClick={(event) => {
                let startOffset = event.startOffset;
                let endOffset = event.endOffset;
                let currSelection = event.selection.substring(startOffset, endOffset);
                let validCurrSelection = currSelection.substring(0, 300);
                validCurrSelection = validCurrSelection.replace(/(?:\r\n|\r|\n)/g, ' ');

                if (role.toLowerCase() == 'domain-expert' && validCurrSelection) {
                    if (validCurrSelection.length < 30) {
                        Popup.create({
                            title: "Too short selection! (min 30 chars)",
                            content: <div> {validCurrSelection} <br/> <br/> <strong> Click anywhere to close! </strong> </div>,
                            buttons: {
                                left: [{
                                    text: 'Close',
                                    className: 'danger',
                                    action: function () {
                                        Popup.close();
                                    }
                                }],
                            }
                            }, true);
                    }
                    else
                    {
                        Popup.create({
                            title: "Your selection is (max 300 chars)",
                            content: <div> {validCurrSelection} <br/> <br/> <strong> Click anywhere to close! </strong> </div>,
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
                                        setSelection(validCurrSelection)
                                        Popup.close();
                                    }
                                }]
                            }
                            }, true);
                    }
                }

            }}
        />
      )}
      onMessagesChange={(messages) => {
        if (messages.length > 0 && 'title' in messages[messages.length - 1] && !title) {
          console.log("setting article data");
          if ('abstract' in messages[messages.length - 1]) {
            setAbstract(messages[messages.length - 1].abstract);
            setIntroduction(messages[messages.length - 1].introduction);
          }
          setTitle(messages[messages.length - 1].title);
        }
        if (messages.length > 0 && 'role' in messages[messages.length - 1]) {
            setRole(messages[messages.length - 1].role);
        }
        if (messages.length > 0 && 'interaction' in messages[messages.length - 1]) {
            setInteractions(messages[messages.length - 1].interaction);
            setMinInteractions(messages[messages.length - 1].min_interactions);
        }
      }}
      selection={selection}
      role={role}
      interactions={interactions}
      minInteractions={minInteractions}
    />
  );
}

ReactDOM.render(<div><MainApp /> <Popup className="mm-popup" btnClass="mm-popup__btn" closeBtn={true} closeHtml={null} defaultOk="Confirm" defaultCancel="Cancel" wildClasses={false} escToClose={true}  /> </div>, document.getElementById("app"));