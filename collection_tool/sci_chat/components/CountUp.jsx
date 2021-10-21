
import React from "react";

function CountUpTimer({ agentStatus, inputMode, hasSent }) {
    const [minutes, setMinutes] = React.useState(0);
    const [seconds, setSeconds] =  React.useState(0);
    const [compAgentStatus, setCompAgentStatus] = React.useState("");
    const [compInputMode, setCompInputMode] = React.useState("");
    const [compHasSent,setCompHasSent] = React.useState(false);
    const [startDate, setStartDate] = React.useState(new Date());
    const [currentDate, setCurrentDate] = React.useState(new Date());

    React.useEffect(() => {
        setCompAgentStatus(agentStatus)
    }, [agentStatus]);

    React.useEffect(() => {
        setCompHasSent(hasSent);
    }, [hasSent]);

    React.useEffect(() => {
        setCompInputMode(inputMode);
    }, [inputMode]);

    React.useEffect(() => {
        if (compHasSent || compInputMode === "waiting") {
            setStartDate(new Date());
            setSeconds(0);
            setMinutes(0);
        }
    }, [compHasSent, compAgentStatus])

    React.useEffect(() => {
    let myInterval = setInterval(() => {
            if (compInputMode === "waiting") {
                setStartDate(new Date());
                setSeconds(0);
                setMinutes(0);
                return
            }

            setCurrentDate(new Date());
            let count = +currentDate - +startDate;

            let seconds = Math.floor((count / 1000)) % 60;
            seconds = Math.max(seconds, 0);
            let minutes = Math.floor((count / 60000)) % 60;
            minutes = Math.max(minutes, 0);

            setSeconds(seconds);
            setMinutes(minutes);

            if (minutes == 60) {
                clearInterval(myInterval)
            }
        }, 1000)
        return ()=> {
            clearInterval(myInterval);
          };
    });

    return (
        <div className="chat-countup-timer">
            <h4> Reply timeout timer (count-up): {minutes}:{seconds < 10 ?  `0${seconds}` : seconds} (04:00)</h4>
        </div>
    )
}

export { CountUpTimer };