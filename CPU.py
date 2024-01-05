import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import deque


class Process:
    def __init__(self, arrival_time, burst_time):
        self.arrival_time = arrival_time
        self.burst_time = burst_time


class Process_S:
    def __init__(self, name, arrival_time, burst_time):
        self.name = name
        self.arrival_time = arrival_time
        self.burst_time = burst_time


class Process_prio:
    def __init__(self, name, burst_time, priority, arrival_time):
        self.name = name
        self.burst_time = burst_time
        self.priority = priority
        self.arrival_time = arrival_time


class Process_rr:
    def __init__(self, pid, burst_time):
        self.pid = pid
        self.burst_time = burst_time


class MLFQ:
    def __init__(self, num_queues):
        self.queues = [[] for _ in range(num_queues)]

    def schedule(self, processes, quantum):
        for process in processes:
            self.queues[0].append(process)  # Add process to the highest priority queue (Queue 0)

        current_queue = 0

        while True:
            if len(self.queues[current_queue]) > 0:
                process = self.queues[current_queue][0]  # Get the first process from the current queue
                print("Running process:", process.name, "from Queue", current_queue, "with burst time:",
                      process.burst_time)

                # Perform the CPU burst for the process (execute until the burst time is fully consumed)
                if process.burst_time <= quantum:
                    process.burst_time = 0
                else:
                    process.burst_time -= quantum

                if process.burst_time > 0:
                    if current_queue + 1 < len(self.queues):
                        self.queues[current_queue + 1].append(
                            process)  # Move the process to the next lower priority queue
                    else:
                        self.queues[current_queue].append(
                            process)  # If already in the lowest priority, stay in the same queue
                else:
                    print("Process", process.name, "completed.")

                self.queues[current_queue].pop(0)  # Remove the executed process from the current queue

            if self._all_queues_empty():
                break
            if len(self.queues[current_queue]) == 0:
                current_queue = (current_queue + 1) % len(self.queues)

    def _all_queues_empty(self):
        for queue in self.queues:
            if len(queue) > 0:
                return False
        return True


def create_processes(num_processes):
    processes = []

    for i in range(num_processes):
        name = f"P{i + 1}"
        arrival_time = int(input(f"Enter arrival time for process {name}: "))
        burst_time = int(input(f"Enter burst time for process {name}: "))
        process = Process_S(name, arrival_time, burst_time)
        processes.append(process)

    return processes


def create_process(num_processes):
    processes = []

    for i in range(num_processes):
        name = f"P{i + 1}"
        burst_time = float(input(f"Enter burst time for process {name}: "))
        priority = float(input(f"Enter priority for process {name}: "))
        arrival_time = float(input(f"Enter arrival time for process {name}: "))
        process = Process_prio(name, burst_time, priority, arrival_time)
        processes.append(process)

    return processes


def fcfs_scheduling(processes):
    processes.sort(key=lambda x: x.arrival_time)  # Sort processes based on arrival time
    n = len(processes)
    current_time = 0
    waiting_time = [0] * n
    turnaround_time = 0
    gantt_chart = []

    for i, process in enumerate(processes):
        if process.arrival_time > current_time:
            current_time = process.arrival_time

        gantt_chart.append((current_time, current_time + process.burst_time))
        waiting_time[i] = current_time - process.arrival_time
        turnaround_time += current_time + process.burst_time - process.arrival_time
        current_time += process.burst_time

    avg_waiting_time = sum(waiting_time) / n
    avg_turnaround_time = turnaround_time / n

    return gantt_chart, avg_waiting_time, avg_turnaround_time, waiting_time


def print_gantt_chart_fcfs(gantt_chart):
    plt.figure(figsize=(10, 5))
    for i, (start_time, end_time) in enumerate(gantt_chart):
        plt.barh(y=i, width=end_time - start_time, left=start_time, height=0.5)

    plt.xlabel("Time")
    plt.ylabel("Processes")
    plt.title("FCFS Scheduling Gantt Chart")
    plt.show()


def print_gantt_chart_rr(gantt_chart):
    plt.figure(figsize=(10, 5))
    for i, (process_name, start_time, end_time) in enumerate(gantt_chart):
        plt.barh(y=i, width=end_time - start_time, left=start_time, height=0.5, label=process_name)

    plt.xlabel("Time")
    plt.ylabel("Processes")
    plt.title("RR Scheduling Gantt Chart")
    plt.legend()
    plt.show()


def print_table(processes):
    headers = ["Process", "Burst Time"]
    table_data = [[process.pid, process.burst_time] for process in processes]

    print("RR Scheduling Table:")
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))


def sjf_scheduling(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.burst_time))
    current_time = 0
    total_processes = len(processes)
    completed_processes = 0
    waiting_time = [0] * total_processes
    turnaround_time = [0] * total_processes
    gantt_chart = []

    while completed_processes < total_processes:
        min_burst_time = float('inf')
        min_burst_index = -1

        for i in range(total_processes):
            if processes[i].arrival_time <= current_time and processes[i].burst_time > 0:
                if processes[i].burst_time < min_burst_time:
                    min_burst_time = processes[i].burst_time
                    min_burst_index = i

        if min_burst_index == -1:
            current_time += 1
            continue

        current_process = processes[min_burst_index]
        current_process_start = current_time
        current_time += current_process.burst_time
        current_process.burst_time = 0

        waiting_time[min_burst_index] = current_time - current_process.arrival_time - current_process_start
        turnaround_time[min_burst_index] = current_time - current_process.arrival_time

        completed_processes += 1
        gantt_chart.append((current_process.name, current_process_start, current_time))

    avg_waiting_time = sum(waiting_time) / total_processes
    avg_turnaround_time = sum(turnaround_time) / total_processes

    print("Non-Preemptive Shortest Job First (SJF) Scheduling:")
    headers = ["Process", "Arrival Time", "Burst Time", "Waiting Time", "Turnaround Time"]
    table_data = [
        [processes[i].name, processes[i].arrival_time, processes[i].burst_time, waiting_time[i], turnaround_time[i]]
        for i in range(total_processes)]

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print(f"\nAverage Waiting Time: {avg_waiting_time}")
    print(f"Average Turnaround Time: {avg_turnaround_time}")

    # Display waiting time for each process
    print("\nWaiting Time for Each Process:")
    for i, process in enumerate(processes):
        print(f"Process {process.name}: Waiting Time = {waiting_time[i]}")

    # Gantt Chart Visualization
    plt.figure(figsize=(10, 5))
    for i, (process_name, start_time, end_time) in enumerate(gantt_chart):
        plt.barh(y=i, width=end_time - start_time, left=start_time, height=0.5, label=process_name)

    plt.xlabel("Time")
    plt.ylabel("Processes")
    plt.title("Non-Preemptive SJF Scheduling Gantt Chart")
    plt.legend()
    plt.show()


def sjf_scheduling_preemptive(processes):
    processes.sort(key=lambda x: (x.arrival_time, x.burst_time))
    current_time = 0
    total_processes = len(processes)
    completed_processes = 0
    waiting_time = [0] * total_processes
    turnaround_time = [0] * total_processes
    remaining_burst = [process.burst_time for process in processes]
    gantt_chart = []

    while completed_processes < total_processes:
        min_burst_time = float('inf')
        min_burst_index = -1

        for i in range(total_processes):
            if processes[i].arrival_time <= current_time and remaining_burst[i] > 0:
                if remaining_burst[i] < min_burst_time:
                    min_burst_time = remaining_burst[i]
                    min_burst_index = i

        if min_burst_index == -1:
            current_time += 1
            continue

        current_process = processes[min_burst_index]
        gantt_chart.append((current_process.name, current_time, current_time + 1))
        remaining_burst[min_burst_index] -= 1

        if remaining_burst[min_burst_index] == 0:
            completed_processes += 1
            waiting_time[min_burst_index] = current_time + 1 - current_process.arrival_time - current_process.burst_time
            turnaround_time[min_burst_index] = current_time + 1 - current_process.arrival_time

        current_time += 1

    avg_waiting_time = sum(waiting_time) / total_processes
    avg_turnaround_time = sum(turnaround_time) / total_processes

    print("Preemptive Shortest Job First (SJF) Scheduling:")
    headers = ["Process", "Arrival Time", "Burst Time", "Waiting Time", "Turnaround Time"]
    table_data = [
        [processes[i].name, processes[i].arrival_time, processes[i].burst_time, waiting_time[i], turnaround_time[i]]
        for i in range(total_processes)]

    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print(f"\nAverage Waiting Time: {avg_waiting_time}")
    print(f"Average Turnaround Time: {avg_turnaround_time}")

    # Display waiting time for each process
    print("\nWaiting Time for Each Process:")
    for i, process in enumerate(processes):
        print(f"Process {process.name}: Waiting Time = {waiting_time[i]}")

    # Gantt Chart Visualization
    plt.figure(figsize=(10, 5))
    for i, (process_name, start_time, end_time) in enumerate(gantt_chart):
        plt.barh(y=i, width=end_time - start_time, left=start_time, height=0.5, label=process_name)

    plt.xlabel("Time")
    plt.ylabel("Processes")
    plt.title("Preemptive SJF Scheduling Gantt Chart")
    plt.legend()
    plt.show()


def priority_scheduling(processes):
    processes.sort(key=lambda x: (x.priority, x.burst_time))
    current_time = 0
    total_processes = len(processes)
    completed_processes = 0
    waiting_time = [0] * total_processes
    turnaround_time = [0] * total_processes
    gantt_chart = []

    while completed_processes < total_processes:
        high_priority = None
        for i in range(total_processes):
            if processes[i].burst_time > 0:
                if high_priority is None or processes[i].priority < high_priority.priority:
                    high_priority = processes[i]

        if high_priority:
            current_process_start = max(current_time, high_priority.arrival_time)
            current_time = current_process_start + high_priority.burst_time
            high_priority.burst_time = 0

            index = processes.index(high_priority)
            waiting_time[index] = current_process_start - high_priority.arrival_time
            turnaround_time[index] = current_time - high_priority.arrival_time

            completed_processes += 1
            gantt_chart.append((high_priority.name, current_process_start, current_time))
        else:
            current_time += 1

    avg_waiting_time = sum(waiting_time) / total_processes
    avg_turnaround_time = sum(turnaround_time) / total_processes

    avg_waiting_time_ms = avg_waiting_time
    avg_turnaround_time_ms = avg_turnaround_time

    print("Priority Scheduling:")
    headers = ["Process", "Burst Time", "Priority", "Waiting Time", "Turnaround Time"]
    table_data = [
        [processes[i].name, processes[i].burst_time, processes[i].priority, waiting_time[i], turnaround_time[i]]
        for i in range(total_processes)]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))
    print(f"\nAverage Waiting Time: {avg_waiting_time_ms:.2f} ms")
    print(f"Average Turnaround Time: {avg_turnaround_time_ms:.2f} ms")

    # Gantt Chart Visualization
    fig, gnt = plt.subplots()
    gnt.set_xlabel('Time')
    gnt.set_ylabel('Processes')

    gnt.set_yticks([i + 1 for i in range(total_processes)])
    gnt.set_yticklabels([f'{processes[i].name}' for i in range(total_processes)])

    for i in range(len(gantt_chart)):
        pname, start, end = gantt_chart[i]
        gnt.broken_barh([(start, end - start)], (i + 0.5, 0.8), facecolors=('blue'))

    plt.title('Priority Scheduling Gantt Chart')
    plt.grid(True)
    plt.show()


def rr_scheduling(processes, time_quantum):
    n = len(processes)
    remaining_time = [process.burst_time for process in processes]
    completed = [False] * n
    current_time = 0
    gantt_chart = []

    queue = deque()
    while True:
        all_processes_completed = True

        for i in range(n):
            if remaining_time[i] > 0:
                all_processes_completed = False

                if remaining_time[i] > time_quantum:
                    gantt_chart.append((processes[i].pid, current_time, current_time + time_quantum))
                    current_time += time_quantum
                    remaining_time[i] -= time_quantum
                else:
                    gantt_chart.append((processes[i].pid, current_time, current_time + remaining_time[i]))
                    current_time += remaining_time[i]
                    remaining_time[i] = 0
                    completed[i] = True

        if all_processes_completed:
            break

    return gantt_chart


def main():
    print("\nChoose a scheduling algorithm:")
    print("1. First-Come, First-Served (FCFS) Scheduling")
    print("2. Non-preemptive Shortest-Job-First (SJF) Scheduling")
    print("3. preemptive Shortest-Job-First (SJF) Scheduling")
    print("4. Priority Scheduling")
    print("5. Round Robin (RR) Scheduling")
    print("6. Multilevel Feedback Queue scheduling")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        num_processes = int(input("Enter the number of processes: "))
        processes = []

        for i in range(num_processes):
            arrival_time = float(input(f"Enter the arrival time for process {i + 1}: "))
            burst_time = float(input(f"Enter the burst time for process {i + 1}: "))
            processes.append(Process(arrival_time, burst_time))

        gantt_chart, avg_waiting_time, avg_turnaround_time, waiting_time = fcfs_scheduling(processes)

        print("\nFCFS Scheduling:")
        print(f"Average Waiting Time: {avg_waiting_time}")
        print(f"Average Turnaround Time: {avg_turnaround_time}")

        print("\nWaiting Time for Each Process:")
        for i, process in enumerate(processes):
            print(f"Process {i + 1}: Waiting Time = {waiting_time[i]}")

        print_gantt_chart_fcfs(gantt_chart)

    elif choice == 2:
        num_processes = int(input("Enter the number of processes: "))
        processes_list = create_processes(num_processes)
        sjf_scheduling(processes_list)

    elif choice == 3:
        num_processes = int(input("Enter the number of processes: "))
        processes_list = create_processes(num_processes)
        sjf_scheduling_preemptive(processes_list)

    elif choice == 4:
        num_processes = int(input("Enter the number of processes: "))
        processes_list = create_process(num_processes)
        priority_scheduling(processes_list)

    elif choice == 5:
        num_processes = int(input("Enter the number of processes: "))
        time_quantum = int(input("Enter the time quantum: "))

        # Get user input for burst time of each process
        processes = []
        for i in range(num_processes):
            burst_time = int(input(f"Enter the burst time for process P{i + 1}: "))
            processes.append(Process_rr(f"P{i + 1}", burst_time))

        gantt_chart = rr_scheduling(processes, time_quantum)
        print_table(processes)
        print_gantt_chart_rr(gantt_chart)

    elif choice == 6:
        num_processes = int(input("Enter the number of processes: "))
        quantum = int(input("Enter the time quantum: "))
        num_queues = int(input("Enter the number of queues: "))

        queues = [[] for _ in range(num_queues)]  # Create queues based on the number entered
        mlfq = MLFQ(num_queues)

        processes = []
        for i in range(num_processes):
            name = f"P{i + 1}"
            arrival_time = int(input(f"Enter arrival time for process {name}: "))
            burst_time = int(input(f"Enter burst time for process {name}: "))
            process = Process_S(name, arrival_time, burst_time)
            processes.append(process)

        mlfq.schedule(processes, quantum)


if __name__ == '__main__':
    main()
