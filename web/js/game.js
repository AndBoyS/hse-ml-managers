// sessionStorage.clear();
// Retrieve state from sessionStorage
let gameState = JSON.parse(sessionStorage.getItem("gameState")) || {
  completedActions: [],
  specialStates: [],
};

const actionsContainer = document.getElementById("actions");
const log = document.getElementById("log");

function createActions() {
  actionsData.forEach((action) => {
    const button = document.createElement("button");
    button.id = `action${action.id}`;
    button.textContent = action.name;
    button.disabled = true;
    actionsContainer.appendChild(button);
  });
}

function isDisabled(prereqs) {
  state = gameState.completedActions + gameState.specialStates;
  prereqOk = prereqs.every((prereq) => state.includes(prereq));
  return !prereqOk;
}

function updateUI() {
  // Enable or disable buttons based on prerequisites
  actionsData.forEach((action) => {
    const button = document.getElementById(`action${action.id}`);
    const disable = isDisabled(action.prerequisited);
    button.disabled = disable;
    if (disable) {
      button.classList.add("action-hidden");
    } else {
      button.classList.add("action-visible");
    }

    if (gameState.completedActions.includes(action.id)) {
      button.classList.add("action-done");
    }
  });

  // Render log
  log.innerHTML = "";
  gameState.completedActions.forEach((actionId) => {
    const li = document.createElement("li");
    li.textContent = actionsData.find((a) => a.id === actionId).name;
    log.appendChild(li);
  });
}

actionsContainer.addEventListener("click", (event) => {
  const button = event.target;
  if (button.tagName === "BUTTON") {
    const actionId = parseInt(button.id.replace("action", ""), 10);
    gameState.completedActions.push(actionId);
    const action = actionsData.filter((x) => x.id == actionId)[0];
    if (action.leads_to) {
      gameState.specialStates.push(action.leads_to);
    }

    sessionStorage.setItem("gameState", JSON.stringify(gameState));
    updateUI();
    console.log(sessionStorage);
  }
});

// Initialize actions and UI
createActions();
updateUI();
