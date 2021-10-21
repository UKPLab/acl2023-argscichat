/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import {
  ConnectionIndicator,
  FormResponse,
  TextResponse,
  SystemMessage,
  DoneResponse,
  ConnectionStatusBoundary,
  ChatPane,
  FactResponse,
  FactGroup,
  CountUpTimer,
  ViewInstructions
} from "../index.js";
import { MephistoContext } from "mephisto-task";
import { AppContext, INPUT_MODE, ChatStatusBar } from "./ChatApp.jsx";

function BaseFrontend({
  messages,
  onMessageSend,
  inputMode,
  renderSidePane,
  renderRightPane,
  renderMessage,
  renderTextResponse,
  renderResponse,
  selection,
  role,
  interactions,
  minInteractions
}) {
  const mephistoContext = React.useContext(MephistoContext);
  const appContext = React.useContext(AppContext);

  const { connectionStatus, agentStatus, taskConfig } = mephistoContext;
  const { appSettings } = appContext;
  const sidePaneSize = appSettings.isCoverPage ? "col-xs-14" : "col-xs-6";

  const [hasSent, setHasSent] = React.useState(false);
  const [sending, setSending] = React.useState(false);

  return (
    <ConnectionStatusBoundary status={connectionStatus}>
      <div className="row" style={{ height: 900}}>

        <div className="right-side-pane">
            {renderRightPane({ mephistoContext, appContext })}
        </div>
        <div className="chat-container-pane">
          <div className="right-top-pane">
            <CountUpTimer agentStatus={agentStatus} hasSent={hasSent} inputMode={inputMode}  />
            <ViewInstructions role={role} />
            <ChatStatusBar className={"chat-status-bar"} />
            <ChatPane scrollBottomKey={messages.length + "-" + inputMode}>
              <div id="message_thread" style={{ width: "100%" }}>
                {messages.map((message, idx) =>
                  renderMessage({ message, idx, appContext, mephistoContext })
                )}
              </div>
              {inputMode === INPUT_MODE.WAITING ? (
                <SystemMessage
                  glyphicon="hourglass"
                  text={getWaitingMessage(agentStatus)}
                />
              ) : null}
            </ChatPane>
          </div>
          {renderResponse ? (
            renderResponse({
              onMessageSend,
              inputMode,
              appContext,
              mephistoContext,
            })
          ) : (
            <div className="response-container">
            <ResponsePane
              inputMode={inputMode}
              onMessageSend={onMessageSend}
              renderTextResponse={renderTextResponse}
              selection={selection}
              role={role}
              interactions={interactions}
              minInteractions={minInteractions}
              setHasSent={setHasSent}
            />
            </div>
          )}
        </div>
      </div>
    </ConnectionStatusBoundary>
  );
}

function getWaitingMessage(agentStatus) {
  return agentStatus === "waiting"
    ? "Waiting to pair with a task..."
    : "Waiting for the next person to speak...";
}

function ResponsePane({ onMessageSend, inputMode, renderTextResponse, selection, role, interactions, minInteractions, setHasSent}) {
  const appContext = React.useContext(AppContext);
  const mephistoContext = React.useContext(MephistoContext);
  const { taskContext, onTaskComplete } = appContext;
  const { agentState = {} } = mephistoContext;

  let response_pane = null;
  switch (inputMode) {
    case INPUT_MODE.DONE:
    case INPUT_MODE.INACTIVE:
      if (!agentState) {
        agentState = {
        "done_text": "Unfortunately, no dialogue partner has connected in time!",
         "task_done": false
         }
      }
      response_pane = (
        <DoneResponse
          onTaskComplete={onTaskComplete}
          onMessageSend={onMessageSend}
          doneText={agentState.done_text || null}
          isTaskDone={agentState.task_done || null}
          agentStatus={mephistoContext.agentStatus}
        />
      );
      break;
    case INPUT_MODE.READY_FOR_INPUT:
    case INPUT_MODE.WAITING:
      if (taskContext && taskContext["respond_with_form"]) {
        response_pane = (
          <FormResponse
            onMessageSend={onMessageSend}
            active={inputMode === INPUT_MODE.READY_FOR_INPUT}
            formOptions={taskContext["respond_with_form"]}
          />
        );
      } else {
        response_pane = renderTextResponse ? (
          renderTextResponse({
            onMessageSend,
            inputMode,
            active: inputMode === INPUT_MODE.READY_FOR_INPUT,
            appContext,
            mephistoContext,
          })
        ) : (
          <FactGroup
            selection={selection}
            role={role}
            interactions={interactions}
            minInteractions={minInteractions}
            onMessageSend={onMessageSend}
            active={inputMode === INPUT_MODE.READY_FOR_INPUT}
            setHasSent={setHasSent}
          />
        );
      }
      break;
    default:
      response_pane = null;
      break;
  }

  return <div className="right-bottom-pane">{response_pane}</div>;
}

function InteractionsViewer( {interactions, maxInteractions} ) {
    const [compInteractions, setCompInteractions] = React.useState(0);
    const [compMaxInteractions, setCompMaxInteractions] = React.useState(0);

    /* Update internal interactions values */
    React.useEffect(() => {
        setCompInteractions(interactions)
    }, [interactions]);

    /* Update internal max interactions values */
    React.useEffect(() => {
        setCompMaxInteractions(maxInteractions)
    }, [maxInteractions]);

    return (
        <div className='chat-interactions-viewer'>
            <h4> Interaction {compInteractions} / {compMaxInteractions} </h4>
        </div>
    );

}

export default BaseFrontend;
