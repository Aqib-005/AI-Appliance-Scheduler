<!DOCTYPE html>
<html>

<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Schedule Appliances</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            overflow-x: hidden;
        }

        h1 {
            margin-bottom: 20px;
        }

        .container {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }

        .appliance-list {
            flex: 1;
            min-width: 250px;
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
        }

        .appliance-list button:hover {
            background-color: #0056b3;
        }

        .weekly-grid {
            flex: 3;
            display: flex;
            gap: 10px;
            overflow-x: auto;
        }

        .day-column {
            flex: 1;
            border: 1px solid #ccc;
            padding: 10px;
            cursor: pointer;
            min-width: 150px;
            background-color: #f9f9f9;
        }

        .day-column.active {
            border: 2px solid #007bff;
            background-color: #e9f5ff;
        }

        .day-column h3 {
            margin-bottom: 10px;
            text-align: center;
        }

        .appliance-item {
            padding: 10px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            border-radius: 5px;
        }

        .appliance-item button {
            border: none;
            padding: 5px 10px;
            cursor: pointer;
            border-radius: 3px;
        }

        .appliance-item button:hover {
            background-color: #cc0000;
        }

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

        .schedule-button {
            margin-top: 20px;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }

        .schedule-button:hover {
            background-color: #218838;
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }

            .weekly-grid {
                flex-direction: column;
            }

            .day-column {
                min-width: 100%;
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
        <!-- Appliance List -->
        <div class="appliance-list">
            <h2>Appliances</h2>
            @foreach ($appliances as $appliance)
                <button
                    onclick="openSchedulePopup('{{ $appliance->id }}', '{{ $appliance->name }}', '{{ $appliance->preferred_start }}', '{{ $appliance->preferred_end }}', '{{ $appliance->duration }}')">
                    {{ $appliance->name }}
                </button>
            @endforeach
        </div>

        <!-- Weekly Grid Table -->
        <div class="weekly-grid">
            @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                <div class="day-column" id="day-{{ strtolower($day) }}" onclick="selectDay('{{ $day }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        @foreach ($selectedAppliances as $appliance)
                            @if (strtolower($appliance->usage_days) === strtolower($day))
                                <div class="appliance-item" id="appliance-{{ $appliance->id }}">
                                    <span>{{ $appliance->name }} ({{ $appliance->preferred_start }} -
                                        {{ $appliance->preferred_end }}, {{ $appliance->duration }} hrs)</span>
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
            <br>
            <label for="preferredStart">Preferred Start Time:</label>
            <input type="time" id="preferredStart" name="preferredStart" required>
            <br>
            <label for="preferredEnd">Preferred End Time:</label>
            <input type="time" id="preferredEnd" name="preferredEnd" required>
            <br>
            <label for="duration">Duration (hours):</label>
            <input type="number" id="duration" name="duration" step="0.01" placeholder="e.g., 3.23" required>
            <br>
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
            <br>
            <label for="editPreferredStart">Preferred Start Time:</label>
            <input type="time" id="editPreferredStart" name="preferredStart" required>
            <br>
            <label for="editPreferredEnd">Preferred End Time:</label>
            <input type="time" id="editPreferredEnd" name="preferredEnd" required>
            <br>
            <label for="editDuration">Duration (hours):</label>
            <input type="number" id="editDuration" name="duration" step="0.01" placeholder="e.g., 2.5" required>
            <br>
            <button type="button" onclick="updateAppliance()">Edit</button>
            <button type="button" onclick="closeEditAppliancePopup()">Cancel</button>
        </div>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        let selectedDay = null;
        let selectedApplianceId = null;
        let selectedApplianceName = null;
        let selectedApplianceStart = null;
        let selectedApplianceEnd = null;
        let selectedApplianceDuration = null;

        // Open the schedule popup
        function openSchedulePopup(applianceId, applianceName, preferredStart, preferredEnd, duration) {
            if (!selectedDay) {
                alert('Please select a day first.');
                return;
            }

            selectedApplianceId = applianceId; // Set the selected appliance ID

            // Fetch the latest appliance data from the database
            fetch(`/appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    // Trim seconds from time values (if present)
                    const preferredStartFormatted = data.preferred_start ? data.preferred_start.substring(0, 5) : '';
                    const preferredEndFormatted = data.preferred_end ? data.preferred_end.substring(0, 5) : '';

                    // Populate the popup form with the latest data
                    document.getElementById('applianceId').value = data.id;
                    document.getElementById('applianceName').value = data.name;
                    document.getElementById('preferredStart').value = preferredStartFormatted;
                    document.getElementById('preferredEnd').value = preferredEndFormatted;
                    document.getElementById('duration').value = data.duration; // Ensure this matches the format
                    document.getElementById('selectedDayName').textContent = selectedDay || '';

                    // Show the popup
                    document.getElementById('schedulePopup').classList.add('active');
                    document.getElementById('overlay').classList.add('active');
                })
                .catch(error => {
                    console.error('Error fetching appliance data:', error);
                    alert('Failed to fetch appliance data. Please try again.');
                });
        }

        // Close the schedule popup
        function closeSchedulePopup() {
            document.getElementById('schedulePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Select a day
        function selectDay(day) {
            // Remove active class from all day columns
            document.querySelectorAll('.day-column').forEach(column => {
                column.classList.remove('active');
            });

            // Add active class to the selected day
            document.getElementById(`day-${day.toLowerCase()}`).classList.add('active');
            selectedDay = day;
        }

        // Run the scheduling algorithm
        function runSchedulingAlgorithm() {
            console.log('Running scheduling algorithm...'); // Debugging

            // Fetch predictions and generate the schedule
            fetch('/schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({}), // No need to send start_date
            })
                .then(response => {
                    console.log('Response status:', response.status);
                    if (!response.ok) {
                        // Log the response text for debugging
                        return response.text().then(text => {
                            console.error('Response text:', text);
                            throw new Error('Network response was not ok');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Schedule generated:', data);
                    if (data.success) {
                        // Redirect to the dashboard
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

        function addApplianceToDay() {
            if (!selectedApplianceId) {
                alert('No appliance selected. Please try again.');
                return;
            }

            const preferredStart = document.getElementById('preferredStart').value;
            const preferredEnd = document.getElementById('preferredEnd').value;
            const duration = parseFloat(document.getElementById('duration').value);

            // Validate inputs
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
                    console.log('API Response:', data); // Debugging
                    if (data.success) {
                        // Create the appliance item
                        const applianceItem = document.createElement('div');
                        applianceItem.className = 'appliance-item';
                        applianceItem.id = `appliance-${data.selected_appliance_id}`; // Set the id attribute
                        applianceItem.innerHTML = `
                    <span>${document.getElementById('applianceName').value} (${preferredStart} - ${preferredEnd}, ${duration} hrs)</span>
                    <button onclick="openEditAppliancePopup('${data.selected_appliance_id}', '${selectedDay}')">Edit</button>
                    <button onclick="removeAppliance(${data.selected_appliance_id}, '${selectedDay.toLowerCase()}')">Remove</button>
                `;

                        // Add the appliance to the selected day's container
                        document.getElementById(`appliances-${selectedDay.toLowerCase()}`).appendChild(applianceItem);

                        // Close the popup
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
                        console.log('Editing appliance ID:', data.appliance.id);
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
            console.log('updateAppliance() triggered');
            const applianceId = document.getElementById('editApplianceId').value;
            console.log('Update URL should be /selected-appliance/update/' + applianceId);
            let preferredStart = document.getElementById('editPreferredStart').value;
            let preferredEnd = document.getElementById('editPreferredEnd').value;
            const duration = parseFloat(document.getElementById('editDuration').value);

            // Trim seconds if present (ensures HH:MM format)
            preferredStart = preferredStart.substring(0, 5);
            preferredEnd = preferredEnd.substring(0, 5);

            // Input validation
            if (!preferredStart || !preferredEnd || isNaN(duration)) {
                alert('Please fill in all fields correctly.');
                return;
            }

            if (duration <= 0) {
                alert('Duration must be greater than 0.');
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
                        applianceElement.querySelector('span').textContent = `${document.getElementById('editApplianceName').value} (${preferredStart} - ${preferredEnd}, ${duration} hrs)`;
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

            // Send a DELETE request to remove the appliance
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
                    console.log('API Response:', data); // Debugging
                    if (data.success) {
                        // Remove the appliance from the UI
                        const applianceElement = document.getElementById(`appliance-${applianceId}`);
                        if (applianceElement) {
                            applianceElement.remove();
                        }
                    } else {
                        alert('Failed to remove appliance. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error); // Debugging
                    alert('Failed to remove appliance. Please try again.');
                });
        }
    </script>
</body>

</html>