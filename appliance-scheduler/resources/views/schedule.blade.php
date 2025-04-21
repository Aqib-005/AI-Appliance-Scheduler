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
            /* Remove horizontal overflow so we can scroll horizontally if needed */
            overflow-x: auto;
        }

        h1 {
            margin-bottom: 20px;
        }

        /* Main container: two columns (left = 10%, right = 90%) */
        .container {
            display: flex;
            flex-wrap: nowrap;
            /* keep columns in one row */
            gap: 20px;
            width: 100%;
        }

        /* Left side: Appliance list => 10% */
        .appliance-list {
            flex: 0 0 10%;
            min-width: 100px;
            /* ensure it's not too skinny on very small screens */
        }

        .appliance-list h2 {
            margin-bottom: 10px;
        }

        .appliance-list button {
            display: block;
            width: 100%;
            padding: 10px;
            margin-bottom: 10px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-align: left;
            background-color: #007bff;
            color: #fff;
        }

        .appliance-list button:hover {
            background-color: #0056b3;
        }

        /* Right side: weekly grid => 90% */
        .weekly-grid {
            flex: 0 0 90%;
            display: flex;
            flex-wrap: nowrap;
            /* columns side by side */
            gap: 10px;
            /* If columns don't fit, scroll horizontally */
            overflow-x: auto;
        }

        .day-column {
            /* Let flex distribute space, but keep a minimum width to prevent vertical text */
            flex: 1 1 auto;
            min-width: 150px;
            border: 1px solid #ccc;
            padding: 10px;
            cursor: pointer;
            background-color: #f9f9f9;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
            /* Ensure the column stretches */
            min-height: 300px;
        }

        .day-column.active {
            border: 2px solid #007bff;
            background-color: #e9f5ff;
        }

        .day-column h3 {
            margin-bottom: 10px;
            text-align: center;
            white-space: nowrap;
            /* keep the day name on one line */
        }

        .appliances-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        /* Appliance items */
        .appliance-item {
            padding: 10px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 5px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            /* Let text and buttons wrap onto multiple lines if needed */
            flex-wrap: wrap;
        }

        /* Prevent text from turning vertical; allow normal wrapping */
        .appliance-item span {
            flex: 1;
            margin-right: 5px;
            word-wrap: break-word;
            white-space: normal;
        }

        .appliance-item button {
            border: none;
            padding: 5px 8px;
            cursor: pointer;
            border-radius: 3px;
            margin-left: 5px;
            background-color: #007bff;
            color: #fff;
            font-size: 0.9em;
        }

        .appliance-item button:hover {
            background-color: #0056b3;
        }

        /* Popups */
        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            border-radius: 5px;
            width: 300px;
        }

        .overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }

        /* Schedule Button */
        .schedule-button {
            margin-top: 20px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            background-color: #218838;
            color: #fff;
        }

        .schedule-button:hover {
            background-color: #1e7e34;
        }

        /* Responsive tweaks for narrower screens */
        @media (max-width: 768px) {
            .container {
                flex-wrap: wrap;
            }

            .appliance-list {
                flex: 1 1 100%;
                min-width: auto;
                margin-bottom: 20px;
            }

            .weekly-grid {
                flex: 1 1 100%;
            }

            .day-column {
                min-width: 200px;
                /* let columns be scrollable horizontally */
            }
        }
    </style>
</head>

<body>
    <h1>Schedule Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button class="schedule-button">Back to Dashboard</button>
    </a>

    <div class="container">
        <!-- Appliance List (10% width) -->
        <div class="appliance-list">
            <h2>Appliances</h2>
            @foreach ($appliances as $appliance)
                <button
                    onclick="openSchedulePopup('{{ $appliance->id }}', '{{ $appliance->name }}', '{{ $appliance->preferred_start }}', '{{ $appliance->preferred_end }}', '{{ $appliance->duration }}')">
                    {{ $appliance->name }}
                </button>
            @endforeach
        </div>

        <div class="weekly-grid">
            @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                <div class="day-column" id="day-{{ strtolower($day) }}" onclick="selectDay('{{ $day }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        @foreach ($selectedAppliances as $appliance)
                            @if (strtolower($appliance->usage_days) === strtolower($day))
                                <div class="appliance-item" id="appliance-{{ $appliance->id }}">
                                    <span>
                                        {{ $appliance->name }}
                                        ({{ $appliance->preferred_start }} - {{ $appliance->preferred_end }},
                                        {{ $appliance->duration }} hrs)
                                    </span>
                                    <button onclick="openEditAppliancePopup('{{ $appliance->id }}', '{{ $day }}')">Edit</button>
                                    <button
                                        onclick="removeAppliance({{ $appliance->id }}, '{{ strtolower($day) }}')">Remove</button>
                                </div>
                            @endif
                        @endforeach
                    </div>
                </div>
            @endforeach
        </div>
    </div>

    <!-- Schedule Button -->
    <button class="schedule-button" onclick="runSchedulingAlgorithm()">Schedule</button>

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
        <div id="editApplianceForm" onsubmit="event.preventDefault();">
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
            <button type="button" onclick="updateAppliance()">Edit</button>
            <button type="button" onclick="closeEditAppliancePopup()">Cancel</button>
        </div>
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
                alert('No appliance selected. Please try again.');
                return;
            }
            const preferredStart = document.getElementById('preferredStart').value;
            const preferredEnd = document.getElementById('preferredEnd').value;
            const duration = parseFloat(document.getElementById('duration').value);

            if (!selectedDay || !preferredStart || !preferredEnd || isNaN(duration)) {
                alert('Please fill in all fields and select a day.');
                return;
            }
            if (duration <= 0) {
                alert('Duration must be a positive number.');
                return;
            }

            fetch('/selected-appliance/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({
                    appliance_id: selectedApplianceId,
                    name: document.getElementById('applianceName').value,
                    preferred_start: preferredStart,
                    preferred_end: preferredEnd,
                    duration: duration,
                    usage_days: selectedDay,
                }),
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        const applianceItem = document.createElement('div');
                        applianceItem.className = 'appliance-item';
                        applianceItem.id = `appliance-${data.selected_appliance_id}`;
                        applianceItem.innerHTML = `
                            <span>
                                ${document.getElementById('applianceName').value}
                                (${preferredStart} - ${preferredEnd}, ${duration} hrs)
                            </span>
                            <button onclick="openEditAppliancePopup('${data.selected_appliance_id}', '${selectedDay}')">Edit</button>
                            <button onclick="removeAppliance(${data.selected_appliance_id}, '${selectedDay.toLowerCase()}')">Remove</button>
                        `;
                        document.getElementById(`appliances-${selectedDay.toLowerCase()}`).appendChild(applianceItem);
                        closeSchedulePopup();
                    } else {
                        alert('Failed to add appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error);
                    alert('Failed to add appliance. Please try again.');
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

        function removeAppliance(applianceId, day) {
            if (!confirm('Are you sure you want to remove this appliance?')) {
                return;
            }
            fetch(`/selected-appliance/remove/${applianceId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        const applianceElement = document.getElementById(`appliance-${applianceId}`);
                        if (applianceElement) {
                            applianceElement.remove();
                        }
                    } else {
                        alert('Failed to remove appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error);
                    alert('Failed to remove appliance. Please try again.');
                });
        }

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