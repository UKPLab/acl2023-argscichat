/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ResizableTextArea from 'react-fluid-textarea';

function SelectableParagraph({text, onClick}) {

    const tryOnClick = React.useCallback(
        (event) => {
            let selection = window.getSelection();
            let range = selection.getRangeAt(0);
            console.log("selection -> ", selection.toString());
            let selectionInfo = {
               "selection": selection.toString(),
               "start": range.startOffset,
               "end": range.endOffset
            }
            onClick(selectionInfo);

        });

    return (
        <div onClick={tryOnClick} className="readable-content"> {text} </div>
    );

}

function Passage({abstract, title, introduction, onClick}) {
    const [abstractData, setAbstractData] = React.useState(null);
    const [introData, setIntroData] = React.useState(null);

  React.useEffect(() => {
     let toRender = [];
     if (abstract) {
        toRender.push(
          <div>
               <h2 className="readable-header">Abstract</h2>
              <div>
                <SelectableParagraph className="readable-content" text={abstract} onClick={onClick} />
              </div>
          </div>
        );
        setAbstractData(toRender);
     }
  }, [abstract]);

    React.useEffect(() => {
     let toRender = [];
     if (introduction) {
        toRender.push(
          <div>
           <h2 className="readable-header">Introduction</h2>
          <div>
          <SelectableParagraph className="readable-content" text={introduction} onClick={onClick} />
          </div>
          </div>
        );
        setIntroData(toRender);
     }
  }, [introduction]);

  if (title) {
    return (<div className="article-section">
      <h2 className="readable-header"> Title </h2>
      <SelectableParagraph className="readable-content" text={title} onClick={onClick} />
      {abstractData}
      {introData}
      </div>
      )
  }
  return null
}

/*
    Imported
*/

function OnboardingComponent({ onSubmit }) {
  return (
    <div>
      <Directions>
        This component only renders if you have chosen to assign an onboarding
        qualification for your task. Click the button to move on to the main
        task.
      </Directions>
      <button
        className="button is-link"
        onClick={() => onSubmit({ success: true })}
      >
        Move to main task.
      </button>
    </div>
  );
}

function LoadingScreen() {
  return <Directions>Loading...</Directions>;
}

function Directions({ children }) {
  return (
    <section className="hero is-light">
      <div className="hero-body">
        <div className="container">
          <p className="subtitle is-5">{children}</p>
        </div>
      </div>
    </section>
  );
}

function SimpleFrontend({ taskData, isOnboarding, onSubmit, onError }) {
  if (!taskData) {
    return <LoadingScreen />;
  }
  if (isOnboarding) {
    return <OnboardingComponent onSubmit={onSubmit} />;
  }
  return (
    <div>
      <Directions>
        Directions: Please rate the below sentence as good or bad.
      </Directions>
      <section className="section">
        <div className="container">
          <p className="subtitle is-5"></p>
          <p className="title is-3 is-spaced">{taskData.text}</p>
          <div className="field is-grouped">
            <div className="control">
              <button
                className="button is-success is-large"
                onClick={() => onSubmit({ rating: "good" })}
              >
                Mark as Good
              </button>
            </div>
            <div className="control">
              <button
                className="button is-danger is-large"
                onClick={() => onSubmit({ rating: "bad" })}
              >
                Mark as Bad
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export {Passage};
