<!DOCTYPE html>
<html>

<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Schedule Appliances</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }

        .header h1 {
            margin: 0;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 1em;
        }

        .btn-primary {
            background-color: #007bff;
            color: #fff;
        }

        .btn-primary:hover {
            background-color: #0056b3;
        }

        .btn-success {
            background-color: #218838;
            color: #fff;
        }

        .btn-success:hover {
            background-color: #1e7e34;
        }

        .container {
            display: flex;
            gap: 20px;
            width: 100%;
        }

        /* Left list */
        .appliance-list {
            flex: 0 0 20%;
            min-width: 150px;
            display: flex;
            flex-direction: column;
        }

        .appliance-list h2 {
            margin-bottom: 10px;
        }

        .appliance-list-items {
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .appliance-list-items button {
            width: 100%;
            padding: 10px;
            border: none;
            border-radius: 5px;
            text-align: left;
            cursor: pointer;
            background: #007bff;
            color: #fff;
            font-size: 1em;
        }

        .appliance-list-items button:hover {
            background: #0056b3;
        }

        .schedule-button {
            margin-top: 10px;
            width: 100%;
        }

        /* Week grid */
        .weekly-grid {
            flex: 1;
            display: flex;
            gap: 10px;
            overflow-x: hidden;
        }

        .day-column {
            flex: 0 0 calc((100% - 60px) / 7);
            padding: 10px;
            background: #f9f9f9;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            min-height: 300px;
        }

        .day-column.active {
            border-color: #007bff;
            background: #e9f5ff;
        }

        .day-column h3 {
            text-align: center;
            white-space: nowrap;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .appliances-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .appliance-item {
            position: relative;
            padding: 10px;
            padding-top: 36px;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 5px;
            display: flex;
            align-items: center;
        }

        .appliance-item span {
            flex: 1;
            word-wrap: break-word;
            line-height: 1.3;
        }

        .action-icons {
            position: absolute;
            top: 6px;
            right: 8px;
            display: flex;
            gap: 6px;
        }

        .icon-btn {
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            color: #007bff;
            padding: 4px;
            line-height: 1;
        }

        .icon-btn:hover {
            color: #0056b3;
        }

        /* Popups & overlay */
        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0, 0, 0, .1);
            z-index: 1000;
            width: 320px;
            max-width: 90%;
        }

        .overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, .5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }

        @media(max-width:768px) {
            .container {
                flex-wrap: wrap;
            }

            .appliance-list {
                width: 100%;
                margin-bottom: 20px;
            }

            .weekly-grid {
                width: 100%;
                overflow-x: auto;
            }

            .day-column {
                flex: 1 1 auto;
                min-width: 200px;
            }
        }
    </style>
</head>

<body>
    <div class="header">
        <h1>Schedule Appliances</h1>
        <a href="{{ route('dashboard') }}">
            <button class="btn btn-success">Back to Dashboard</button>
        </a>
    </div>

    <div class="container">
        <!-- Left column: appliances + schedule -->
        <div class="appliance-list">
            <h2>Appliances</h2>
            <div class="appliance-list-items">
                @foreach($appliances as $appliance)
                    <button onclick="openSchedulePopup('{{ $appliance->id }}')">
                        {{ $appliance->name }}
                    </button>
                @endforeach
            </div>
            <button class="btn btn-success schedule-button" onclick="runSchedulingAlgorithm()">
                Schedule
            </button>
        </div>

        <!-- Right column: week grid -->
        <div class="weekly-grid">
            @foreach(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                <div class="day-column" id="day-{{ strtolower($day) }}" onclick="selectDay('{{ $day }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        @foreach($selectedAppliances as $appl)
                            @if(strtolower($appl->usage_days) === strtolower($day))
                                <div class="appliance-item" id="appliance-{{ $appl->id }}">
                                    <div class="action-icons">
                                        <button class="icon-btn" onclick="openEditAppliancePopup('{{ $appl->id }}','{{ $day }}')"
                                            title="Edit">&#9998;</button>
                                        <button class="icon-btn"
                                            onclick="openRemoveConfirmation('{{ $appl->id }}','{{ strtolower($day) }}')"
                                            title="Remove">&#128465;</button>
                                    </div>
                                    <span>
                                        {{ $appl->name }}
                                        ({{ $appl->preferred_start }} - {{ $appl->preferred_end }},
                                        {{ $appl->duration }}h)
                                    </span>
                                </div>
                            @endif
                        @endforeach
                    </div>
                </div>
            @endforeach
        </div>
    </div>

    <!-- Schedule Popup -->
    <div id="schedulePopup" class="popup">
        <h2>Schedule Appliance for <span id="selectedDayName"></span></h2>
        <form id="scheduleForm">
            <input type="hidden" id="applianceId" name="applianceId">
            <label for="applianceName">Appliance:</label>
            <input type="text" id="applianceName" name="applianceName" readonly>
            <br><br>
            <label for="preferredStart">Preferred Start Time:</label>
            <input type="time" id="preferredStart" name="preferredStart" required>
            <br><br>
            <label for="preferredEnd">Preferred End Time:</label>
            <input type="time" id="preferredEnd" name="preferredEnd" required>
            <br><br>
            <label for="duration">Duration (hours):</label>
            <input type="number" id="duration" name="duration" step="0.01" placeholder="e.g., 3.23" required>
            <br><br>
            <button type="button" onclick="addApplianceToDay()">Add</button>
            <button type="button" onclick="closeSchedulePopup()">Cancel</button>
        </form>
    </div>

    <!-- Edit Appliance Popup -->
    <div id="editAppliancePopup" class="popup">
        <h2>Edit Appliance for <span id="editSelectedDayName"></span></h2>
        <form onsubmit="event.preventDefault();">
            <input type="hidden" id="editApplianceId" name="applianceId">
            <label for="editApplianceName">Appliance:</label>
            <input type="text" id="editApplianceName" name="applianceName" readonly>
            <br><br>
            <label for="editPreferredStart">Preferred Start Time:</label>
            <input type="time" id="editPreferredStart" name="preferredStart" required>
            <br><br>
            <label for="editPreferredEnd">Preferred End Time:</label>
            <input type="time" id="editPreferredEnd" name="preferredEnd" required>
            <br><br>
            <label for="editDuration">Duration (hours):</label>
            <input type="number" id="editDuration" name="duration" step="0.01" placeholder="e.g., 2.5" required>
            <br><br>
            <button type="button" onclick="updateAppliance()">Save</button>
            <button type="button" onclick="closeEditAppliancePopup()">Cancel</button>
        </form>
    </div>

    <!-- Remove Confirmation Popup -->
    <div id="removeConfirmationPopup" class="popup">
        <h2>Confirm Removal</h2>
        <p>Are you sure you want to remove this appliance?</p>
        <button type="button" id="confirmRemoveBtn">Confirm</button>
        <button type="button" onclick="closeRemoveConfirmation()">Cancel</button>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        let selectedDay = null;
        let selectedApplianceId = null;

        function selectDay(day) {
            document.querySelectorAll('.day-column').forEach(column => {
                column.classList.remove('active');
            });
            document.getElementById(`day-${day.toLowerCase()}`).classList.add('active');
            selectedDay = day;
        }

        function openSchedulePopup(applianceId, applianceName, preferredStart, preferredEnd, duration) {
            if (!selectedDay) {
                alert('Please select a day first.');
                return;
            }
            selectedApplianceId = applianceId;

            fetch(`/appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    const preferredStartFormatted = data.preferred_start ? data.preferred_start.substring(0, 5) : '';
                    const preferredEndFormatted = data.preferred_end ? data.preferred_end.substring(0, 5) : '';

                    document.getElementById('applianceId').value = data.id;
                    document.getElementById('applianceName').value = data.name;
                    document.getElementById('preferredStart').value = preferredStartFormatted;
                    document.getElementById('preferredEnd').value = preferredEndFormatted;
                    document.getElementById('duration').value = data.duration;
                    document.getElementById('selectedDayName').textContent = selectedDay || '';

                    document.getElementById('schedulePopup').classList.add('active');
                    document.getElementById('overlay').classList.add('active');
                })
                .catch(error => {
                    console.error('Error fetching appliance data:', error);
                    alert('Failed to fetch appliance data. Please try again.');
                });
        }

        function closeSchedulePopup() {
            document.getElementById('schedulePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        function addApplianceToDay() {
            if (!selectedApplianceId) {
                alert('No appliance selected.');
                return;
            }
            const start = document.getElementById('preferredStart').value,
                end = document.getElementById('preferredEnd').value,
                dur = parseFloat(document.getElementById('duration').value);
            if (!selectedDay || !start || !end || isNaN(dur) || dur <= 0) {
                alert('Please fill all fields and select a day.');
                return;
            }

            fetch('/selected-appliance/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({
                    appliance_id: selectedApplianceId,
                    name: document.getElementById('applianceName').value,
                    preferred_start: start,
                    preferred_end: end,
                    duration: dur,
                    usage_days: selectedDay
                })
            })
                .then(r => r.json())
                .then(data => {
                    if (!data.success) throw new Error();
                    // build exactly the same markup as server-rendered
                    const item = document.createElement('div');
                    item.className = 'appliance-item';
                    item.id = `appliance-${data.selected_appliance_id}`;
                    item.innerHTML = `
                    <div class="action-icons">
                      <button class="icon-btn"
                        onclick="openEditAppliancePopup('${data.selected_appliance_id}','${selectedDay}')"
                        title="Edit">&#9998;</button>
                      <button class="icon-btn"
                        onclick="openRemoveConfirmation('${data.selected_appliance_id}','${selectedDay.toLowerCase()}')"
                        title="Remove">&#128465;</button>
                    </div>
                    <span>
                      ${document.getElementById('applianceName').value}
                      (${start} - ${end}, ${dur.toFixed(2)}h)
                    </span>
                `;
                    document
                        .getElementById(`appliances-${selectedDay.toLowerCase()}`)
                        .appendChild(item);
                    closeSchedulePopup();
                })
                .catch(() => {
                    alert('Failed to add appliance.');
                });
        }

        function openEditAppliancePopup(applianceId, day) {
            fetch(`/selected-appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('editApplianceId').value = data.appliance.id;
                        document.getElementById('editApplianceName').value = data.appliance.name;
                        document.getElementById('editPreferredStart').value = data.appliance.preferred_start;
                        document.getElementById('editPreferredEnd').value = data.appliance.preferred_end;
                        document.getElementById('editDuration').value = data.appliance.duration;
                        document.getElementById('editSelectedDayName').textContent = day;

                        document.getElementById('editAppliancePopup').classList.add('active');
                        document.getElementById('overlay').classList.add('active');
                    } else {
                        alert('Failed to load appliance details.');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while loading appliance details.');
                });
        }

        function updateAppliance() {
            const applianceId = document.getElementById('editApplianceId').value;
            let preferredStart = document.getElementById('editPreferredStart').value;
            let preferredEnd = document.getElementById('editPreferredEnd').value;
            const duration = parseFloat(document.getElementById('editDuration').value);

            preferredStart = preferredStart.substring(0, 5);
            preferredEnd = preferredEnd.substring(0, 5);

            if (!preferredStart || !preferredEnd || isNaN(duration) || duration <= 0) {
                alert('Please fill in all fields correctly.');
                return;
            }

            fetch(`/selected-appliance/update/${applianceId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content,
                },
                body: JSON.stringify({
                    preferred_start: preferredStart,
                    preferred_end: preferredEnd,
                    duration: duration,
                }),
            })
                .then(response => {
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error(`HTTP error! Status: ${response.status}, Response: ${text}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        const applianceElement = document.getElementById(`appliance-${applianceId}`);
                        applianceElement.querySelector('span').textContent =
                            `${document.getElementById('editApplianceName').value} (${preferredStart} - ${preferredEnd}, ${duration} hrs)`;
                        closeEditAppliancePopup();
                    } else {
                        alert('Failed to update appliance.');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while saving changes: ' + error.message);
                });
        }

        function closeEditAppliancePopup() {
            document.getElementById('editAppliancePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        function openRemoveConfirmation(id, day) {
            pendingRemoveId = id;
            pendingRemoveDay = day;
            document.getElementById('removeConfirmationPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeRemoveConfirmation() {
            document.getElementById('removeConfirmationPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }
        document.getElementById('confirmRemoveBtn').addEventListener('click', () => {
            fetch(`/selected-appliance/remove/${pendingRemoveId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                }
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById(`appliance-${pendingRemoveId}`).remove();
                    } else {
                        alert('Remove failed');
                    }
                })
                .finally(closeRemoveConfirmation);
        });

        function runSchedulingAlgorithm() {
            fetch('/schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({}),
            })
                .then(response => {
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error('Network response was not ok');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        window.location.href = data.redirect_url;
                    } else {
                        alert('Failed to generate schedule. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error);
                    alert('Failed to generate schedule. Please try again.');
                });
        }
    </script>
</body>

</html>