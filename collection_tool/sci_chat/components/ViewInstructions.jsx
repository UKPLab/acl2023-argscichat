
import React from "react";
import Popup from "react-popup";
import { Button } from "react-bootstrap";
import { DefaultTaskDescription } from "./DefaultTaskDescription.jsx";

function ViewInstructions({ role }) {
    const [roleToShow, setRoleToShow] = React.useState("");

    React.useEffect(
    () => {
        if (role.toLowerCase() == 'domain-expert')
        {
            setRoleToShow("Domain-Expert");
        }
        else if (role.toLowerCase() == 'proponent')
        {
            setRoleToShow("Proponent");
        }
        else
        {
            setRoleToShow("");
        }

    }, [role]);

    const tryView = React.useCallback(
        () => {
            Popup.create({
            title: roleToShow + " Instructions",
            content: <div> <DefaultTaskDescription role={role}></DefaultTaskDescription> </div>,
            buttons: {
                right: [{
                    text: 'Ok',
                    className: 'success',
                    action: function() {
                        Popup.close();
                    }
                }]
            }
        }, true);

    });

    return (
        <div className="chat-view-instructions">
        <Button
            className="btn btn-primary view-instructions"
            onClick={() => tryView()}
        >
        View Instructions
        </Button>
       </div>
    )

}

export { ViewInstructions };