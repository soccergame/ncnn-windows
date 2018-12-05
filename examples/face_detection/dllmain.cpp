// dllmain.cpp : Defines the entry point for the DLL application.
//#include "stdafx.h"
#include "stdio.h"
#include "stdlib.h"

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
extern char g_szFaceDetectionDLLPath[_MAX_PATH];

BOOL APIENTRY DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
					 )
{
	char *pDest = NULL;
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
        GetModuleFileNameA(hModule, g_szFaceDetectionDLLPath, _MAX_PATH);
        pDest = strrchr(g_szFaceDetectionDLLPath, '\\');
		pDest[1] = '\0';

	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}
#endif	// WIN32
