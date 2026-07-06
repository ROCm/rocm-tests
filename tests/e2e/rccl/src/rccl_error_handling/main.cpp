/*
   Copyright © Advanced Micro Devices, Inc., or its affiliates.
   SPDX-License-Identifier:  MIT
*/

/*
 * RCCL error-handling / crash-path probe.
 *
 * Initializes a single-rank RCCL communicator, then deliberately dereferences
 * a null pointer to raise SIGSEGV. RCCL's *own* signal handler (controlled by
 * the RCCL_ENABLE_SIGNALHANDLER env var) decides whether the fault is
 * intercepted and logged ("Inside handler function signal") or delivered
 * straight to the OS. The Python test toggles that env var and asserts the
 * presence/absence of RCCL's handler output — so this stub must NOT install a
 * handler of its own.
 */

#include <rccl/rccl.h>

int main() {
  ncclComm_t comm;
  ncclCommInitAll(&comm, 1, NULL);

  int* ptr = nullptr;
  *ptr = 5;

  return 0;
}
