//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <record_windows/record_windows_plugin_c_api.h>
#include <serious_python_windows/serious_python_windows_plugin_c_api.h>

void RegisterPlugins(flutter::PluginRegistry* registry) {
  RecordWindowsPluginCApiRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("RecordWindowsPluginCApi"));
  SeriousPythonWindowsPluginCApiRegisterWithRegistrar(
      registry->GetRegistrarForPlugin("SeriousPythonWindowsPluginCApi"));
}
