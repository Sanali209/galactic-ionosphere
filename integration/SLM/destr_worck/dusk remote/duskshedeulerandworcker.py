from distributed import Scheduler, Worker
import asyncio

#ngrok tcp 8786
async def start_scheduler_and_worker():
    # Start a Dask scheduler
    scheduler = await Scheduler(host="0.0.0.0", port=8786)  # Bind scheduler to all IPs
    print("Scheduler address:", scheduler.address)

    # Start a worker connected to the scheduler
    #worker = await Worker("tcp://4.tcp.eu.ngrok.io:16404", nthreads=1,memory_limit=10.5e9)  # 1 thread, 6.5GB memory
    #print("Worker connected to:", scheduler.address)

    await scheduler.finished()
    #await worker.finished()


# Run the async function
asyncio.run(start_scheduler_and_worker())