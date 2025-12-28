#include "nperf/output/formatter.h"
#include "nperf/output/json_formatter.h"
#include "nperf/output/text_formatter.h"

namespace nperf {

std::unique_ptr<Formatter> Formatter::create(OutputFormat format) {
    return createFormatter(format);
}

std::unique_ptr<Formatter> createFormatter(OutputFormat format) {
    switch (format) {
        case OutputFormat::JSON:
        case OutputFormat::JSONPretty:
            return std::make_unique<JsonFormatter>(format == OutputFormat::JSONPretty);
        case OutputFormat::Text:
        default:
            return std::make_unique<TextFormatter>();
    }
}

} // namespace nperf
